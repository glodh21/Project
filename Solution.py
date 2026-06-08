import numpy as np
import cv2
import pywt
import os
from canny import CannyEdgeDetector
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------
# Threshold schedule — stricter at coarser levels
# ---------------------------------------------------------------

BASE_LOW    = 0.04
BASE_HIGH   = 0.15
SCALE_FACTOR = 1.07  # reduced from 1.2 to avoid over-strict coarse thresholds

def get_thresholds(level):
    """level is 1-indexed. Returns (low, high) for that DWT level."""
    factor = SCALE_FACTOR ** (level - 1)
    return BASE_LOW * factor, BASE_HIGH * factor


# ---------------------------------------------------------------
# Parent-seeded hysteresis with distance transform
# ---------------------------------------------------------------

def seeded_hysteresis(suppressed, low_thresh, high_thresh, parent_edges=None):
    from collections import deque

    edges   = np.zeros_like(suppressed, dtype=np.uint8)
    strong  = suppressed >= high_thresh
    weak    = (suppressed >= low_thresh) & ~strong

    if parent_edges is not None:
        parent_binary = parent_edges > 0
        dist = distance_transform_edt(~parent_binary)
        parent_mask = dist <= 3.0  # increased from 2.0 for better coverage
        seed_mask = strong | parent_mask
    else:
        seed_mask = strong

    edges[seed_mask] = 255
    visited = seed_mask.copy()
    queue   = deque(zip(*np.where(seed_mask)))

    neighbors = [(-1,-1),(-1, 0),(-1, 1),
                 ( 0,-1),        ( 0, 1),
                 ( 1,-1),( 1, 0),( 1, 1)]

    h, w = edges.shape
    while queue:
        x, y = queue.popleft()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                if not visited[nx, ny] and weak[nx, ny]:
                    edges[nx, ny] = 255
                    visited[nx, ny] = True
                    queue.append((nx, ny))

    # Morphological closing — increased to 5x5 to bridge larger gaps at L1->L2
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    return edges


# ---------------------------------------------------------------
# Shared Gradient Pyramid (downsample L1 results)
# ---------------------------------------------------------------

def build_gradient_pyramid(finest_subband, num_levels, detector):
    sb = finest_subband.astype(np.float32)
    sb_min, sb_max = sb.min(), sb.max()
    if sb_max - sb_min < 1e-8:
        h, w = sb.shape
        grad_mags       = []
        suppressed_maps = []
        for l in range(num_levels):
            scale = 2 ** l
            gh, gw = max(1, h // scale), max(1, w // scale)
            grad_mags.append(np.zeros((gh, gw), dtype=np.float32))
            suppressed_maps.append(np.zeros((gh, gw), dtype=np.float32))
        return grad_mags, suppressed_maps

    sb_norm = (sb - sb_min) / (sb_max - sb_min)

    smoothed                 = detector.gaussian_smooth(sb_norm)
    grad_mag_l1, grad_dir_l1 = detector.compute_gradients(smoothed)
    suppressed_l1            = detector.non_maximum_suppression(grad_mag_l1, grad_dir_l1)

    grad_mags       = [grad_mag_l1]
    suppressed_maps = [suppressed_l1]

    prev_h, prev_w = smoothed.shape[:2]
    for l in range(1, num_levels):
        new_h = max(1, prev_h // 2)
        new_w = max(1, prev_w // 2)

        # Gaussian blur before downsampling to preserve edge presence
        blurred_grad      = cv2.GaussianBlur(grad_mag_l1,   (3, 3), 0)
        blurred_suppressed = cv2.GaussianBlur(suppressed_l1, (3, 3), 0)

        grad_mag   = cv2.resize(blurred_grad,       (new_w, new_h), interpolation=cv2.INTER_AREA)
        suppressed = cv2.resize(blurred_suppressed, (new_w, new_h), interpolation=cv2.INTER_AREA)

        grad_mags.append(grad_mag)
        suppressed_maps.append(suppressed)

        prev_h, prev_w = new_h, new_w

    return grad_mags, suppressed_maps


# ---------------------------------------------------------------
# Persistence-Based Filtering (applied BEFORE hysteresis)
# ---------------------------------------------------------------

def apply_persistence_to_gradient(suppressed, fine_edge_map, target_size, neighborhood=4):
    """
    Zero out gradient magnitudes where fine level has no edge within neighborhood.
    neighborhood increased to 4 (from 2) for L1->L2 scale jump tolerance.
    """
    if fine_edge_map is None:
        return suppressed

    kernel_size = 2 * neighborhood + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    fine_dilated = cv2.dilate(fine_edge_map, kernel, iterations=1)

    fine_dilated_resized = cv2.resize(
        fine_dilated,
        target_size,
        interpolation=cv2.INTER_NEAREST
    )

    suppressed_filtered = suppressed.copy()
    suppressed_filtered[fine_dilated_resized == 0] = 0

    return suppressed_filtered


# ---------------------------------------------------------------
# Scale-consistent edge detection for one subband pair
# ---------------------------------------------------------------

def detect_nested_edges(subbands, subband_name):
    """
    subbands: list of numpy arrays [level1, level2, level3]
              each is a float32 DWT subband (not uint8)
    subband_name: string like 'HL', 'LH', 'HH' — only for printing

    Returns: list of edge maps [E1, E2, E3] as uint8, nested guaranteed.
    """
    num_levels = len(subbands)
    edge_maps  = []

    detector_for_pyramid = CannyEdgeDetector(sigma=1.4, gaussian_size=5)
    grad_mags, suppressed_maps = build_gradient_pyramid(
        subbands[0], num_levels, detector_for_pyramid
    )

    for i in range(num_levels):
        level = i + 1
        low, high = get_thresholds(level)

        subband = subbands[i]

        sb = subband.astype(np.float32)
        if sb.max() - sb.min() < 1e-8:
            edge_maps.append(np.zeros(sb.shape, dtype=np.uint8))
            continue

        suppressed = suppressed_maps[i]

        target_h, target_w = subband.shape[:2]
        if suppressed.shape != (target_h, target_w):
            suppressed = cv2.resize(
                suppressed, (target_w, target_h),
                interpolation=cv2.INTER_LINEAR
            )

        # Persistence filtering BEFORE hysteresis
        if i > 0:
            suppressed = apply_persistence_to_gradient(
                suppressed=suppressed,
                fine_edge_map=edge_maps[i - 1],
                target_size=(target_w, target_h),
                neighborhood=4   # increased from 2
            )

        # Prepare parent seeds — increased dilation iterations from 1 to 3
        parent_edges_downsampled = None
        if i > 0:
            parent_edge_map = edge_maps[i - 1]
            parent_edges_downsampled = cv2.resize(
                parent_edge_map,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR
            )
            kernel = np.ones((3, 3), np.uint8)
            parent_edges_downsampled = cv2.dilate(parent_edges_downsampled, kernel, iterations=3)  # was 1

        edges = seeded_hysteresis(suppressed, low, high, parent_edges_downsampled)

        edge_maps.append(edges)
        print(f"  [{subband_name} Level {level}] thresholds=({low:.4f},{high:.4f})  "
              f"edges={int((edges > 0).sum())}")

    return edge_maps


# ---------------------------------------------------------------
# Downsample fine edge maps to each coarse level's native size
# ---------------------------------------------------------------

def downsample_to_native(edge_maps):
    return edge_maps  # already native resolution


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    import time

    IMAGE_PATH  = 'lana.png'
    OUTPUT_DIR  = 'Sol'
    WAVELET     = 'haar'
    NUM_LEVELS  = 3

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: could not read {IMAGE_PATH}")
        exit()

    cv2.imwrite(os.path.join(OUTPUT_DIR, 'original.jpg'), image)

    print("Running DWT decomposition...")
    coeffs = pywt.wavedec2(image.astype(np.float32), wavelet=WAVELET, level=NUM_LEVELS)
    detail_levels = list(reversed(coeffs[1:]))   # [details_L1, details_L2, details_L3]

    subband_names = ['HL', 'LH', 'HH']
    for level_idx, details in enumerate(detail_levels):
        level = level_idx + 1
        for sb_idx, (name, sb) in enumerate(zip(subband_names, details)):
            sb_vis = sb.astype(np.float32)
            sb_vis = ((sb_vis - sb_vis.min()) / (sb_vis.max() - sb_vis.min() + 1e-8) * 255).astype(np.uint8)
            fname  = os.path.join(OUTPUT_DIR, f'{name}_L{level}_subband.jpg')
            cv2.imwrite(fname, sb_vis)

    print("DWT subbands saved.\n")

    t0 = time.time()
    all_results = {}

    for sb_idx, name in enumerate(subband_names):
        print(f"Processing {name} subbands...")
        subbands_this_type = [detail_levels[l][sb_idx] for l in range(NUM_LEVELS)]
        edge_maps = detect_nested_edges(subbands_this_type, name)
        all_results[name] = edge_maps

        for level_idx, em in enumerate(edge_maps):
            level = level_idx + 1
            fname = os.path.join(OUTPUT_DIR, f'{name}_L{level}_edges_native.jpg')
            cv2.imwrite(fname, em)
            print(f"  Saved {fname}  size={em.shape}")

        h2, w2 = edge_maps[1].shape[:2]
        fine_at_l2 = cv2.resize(edge_maps[0], (w2, h2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'{name}_L1_downsampled_to_L2.jpg'), fine_at_l2)

        if NUM_LEVELS >= 3:
            h3, w3 = edge_maps[2].shape[:2]
            fine_at_l3 = cv2.resize(edge_maps[0], (w3, h3), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{name}_L1_downsampled_to_L3.jpg'), fine_at_l3)
            l2_at_l3 = cv2.resize(edge_maps[1], (w3, h3), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{name}_L2_downsampled_to_L3.jpg'), l2_at_l3)

        print()

    print(f"All done in {time.time() - t0:.4f}s")
    print(f"All outputs saved to '{OUTPUT_DIR}/'")
    print()
    print("To validate consistency, run:")
    print(f"  python3 upscale.py --fine Sol/HL_L1_edges_native.jpg --coarse Sol/HL_L2_edges_native.jpg")
    print(f"  python3 upscale.py --fine Sol/HL_L2_edges_native.jpg --coarse Sol/HL_L3_edges_native.jpg")