import numpy as np
import cv2
import pywt
import os
from canny import CannyEdgeDetector  # your existing canny file


# ---------------------------------------------------------------
# Threshold schedule — stricter at coarser levels (Solution 4)
# ---------------------------------------------------------------

BASE_LOW    = 0.04
BASE_HIGH   = 0.10
SCALE_FACTOR = 1.2

def get_thresholds(level):
    """level is 1-indexed. Returns (low, high) for that DWT level."""
    factor = SCALE_FACTOR ** (level - 1)
    return BASE_LOW * factor, BASE_HIGH * factor


# ---------------------------------------------------------------
# Parent-seeded hysteresis (Solution 2)
# ---------------------------------------------------------------

def seeded_hysteresis(suppressed, low_thresh, high_thresh, parent_edges=None):
    """
    Modified hysteresis that seeds BFS from parent edge map instead of
    (or in addition to) strong pixels.

    parent_edges: binary uint8 map (255=edge, 0=no edge) already downsampled
                  to the same size as suppressed. If None, runs standard hysteresis.
    """
    from collections import deque

    edges   = np.zeros_like(suppressed, dtype=np.uint8)
    strong  = suppressed >= high_thresh
    weak    = (suppressed >= low_thresh) & ~strong

    # seed map: strong pixels from own gradient PLUS parent edge positions
    if parent_edges is not None:
        parent_mask = parent_edges > 0
        seed_mask   = strong | parent_mask
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

    return edges


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

    for i, subband in enumerate(subbands):
        level = i + 1
        low, high = get_thresholds(level)

        # convert subband to uint8 for feeding into CannyEdgeDetector
        # (your Canny handles float input too, but DWT coefficients can be
        #  negative — shift and scale to [0,255] safely)
        sb = subband.astype(np.float32)
        sb_min, sb_max = sb.min(), sb.max()
        if sb_max - sb_min < 1e-8:
            # flat subband — no edges
            edge_maps.append(np.zeros(sb.shape, dtype=np.uint8))
            continue

        sb_norm = ((sb - sb_min) / (sb_max - sb_min) * 255).astype(np.uint8)

        # use your CannyEdgeDetector for smoothing, gradient, NMS
        detector = CannyEdgeDetector(
            sigma=1.4,
            low_threshold=low,
            high_threshold=high,
            gaussian_size=5
        )

        # run only up to NMS — we do hysteresis ourselves
        if len(sb_norm.shape) == 3:
            sb_norm = cv2.cvtColor(sb_norm, cv2.COLOR_BGR2GRAY)

        img_float = sb_norm.astype(np.float32) / 255.0
        smoothed  = detector.gaussian_smooth(img_float)
        grad_mag, grad_dir = detector.compute_gradients(smoothed)
        suppressed = detector.non_maximum_suppression(grad_mag, grad_dir)

        # prepare parent seeds for levels > 1
        parent_edges_downsampled = None
        if i > 0:
            parent_edge_map = edge_maps[i - 1]           # E_{L-1}, full size of level L-1
            target_h, target_w = subband.shape[:2]
            # downsample parent to current level size
            parent_edges_downsampled = cv2.resize(
                parent_edge_map,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            )

        # run modified hysteresis
        edges = seeded_hysteresis(suppressed, low, high, parent_edges_downsampled)

        # safety AND — hard guarantee nesting
        if parent_edges_downsampled is not None:
            edges = cv2.bitwise_and(edges, parent_edges_downsampled)

        edge_maps.append(edges)
        print(f"  [{subband_name} Level {level}] thresholds=({low:.4f},{high:.4f})  "
              f"edges={int((edges > 0).sum())}")

    return edge_maps


# ---------------------------------------------------------------
# Upsample edge maps back to level 1 size for saving / validation
# ---------------------------------------------------------------

def upsample_to_level1(edge_maps):
    """
    Upsample all edge maps to the size of edge_maps[0] (level 1).
    Returns list of same-size uint8 maps.
    """
    h, w    = edge_maps[0].shape[:2]
    upsampled = [edge_maps[0].copy()]
    for em in edge_maps[1:]:
        up = cv2.resize(em, (w, h), interpolation=cv2.INTER_NEAREST)
        upsampled.append(up)
    return upsampled


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    import time

    IMAGE_PATH  = 'lana.png'           # change to your input image
    OUTPUT_DIR  = 'Sol'
    WAVELET     = 'haar'
    NUM_LEVELS  = 3

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- load image ---
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: could not read {IMAGE_PATH}")
        exit()

    cv2.imwrite(os.path.join(OUTPUT_DIR, 'original.jpg'), image)

    # --- DWT decomposition ---
    print("Running DWT decomposition...")
    coeffs = pywt.wavedec2(image.astype(np.float32), wavelet=WAVELET, level=NUM_LEVELS)
    # coeffs[0]         = LL_N  (approximation at coarsest level)
    # coeffs[1]         = (HL_N, LH_N, HH_N)
    # coeffs[2]         = (HL_{N-1}, LH_{N-1}, HH_{N-1})
    # coeffs[3]         = (HL_1, LH_1, HH_1)   ← finest

    # rearrange so index 0 = level 1 (finest), index 2 = level 3 (coarsest)
    detail_levels = list(reversed(coeffs[1:]))   # [details_L1, details_L2, details_L3]

    # save raw DWT subbands as images
    subband_names = ['HL', 'LH', 'HH']
    for level_idx, details in enumerate(detail_levels):
        level = level_idx + 1
        for sb_idx, (name, sb) in enumerate(zip(subband_names, details)):
            sb_vis = sb.astype(np.float32)
            sb_vis = ((sb_vis - sb_vis.min()) / (sb_vis.max() - sb_vis.min() + 1e-8) * 255).astype(np.uint8)
            fname = os.path.join(OUTPUT_DIR, f'{name}_L{level}_subband.jpg')
            cv2.imwrite(fname, sb_vis)

    print("DWT subbands saved.\n")

    # --- nested edge detection per subband type ---
    t0 = time.time()

    all_results = {}   # key: 'HL' / 'LH' / 'HH', value: [E1, E2, E3]

    for sb_idx, name in enumerate(subband_names):
        print(f"Processing {name} subbands...")
        subbands_this_type = [detail_levels[l][sb_idx] for l in range(NUM_LEVELS)]
        edge_maps = detect_nested_edges(subbands_this_type, name)
        all_results[name] = edge_maps

        # save raw edge maps at native resolution
        for level_idx, em in enumerate(edge_maps):
            level = level_idx + 1
            fname = os.path.join(OUTPUT_DIR, f'{name}_L{level}_edges.jpg')
            cv2.imwrite(fname, em)

        # upsample all levels to level 1 size and save
        upsampled = upsample_to_level1(edge_maps)
        for level_idx, up in enumerate(upsampled):
            level = level_idx + 1
            fname = os.path.join(OUTPUT_DIR, f'{name}_L{level}_edges_upsampled.jpg')
            cv2.imwrite(fname, up)

        print()

    print(f"All done in {time.time() - t0:.4f}s")
    print(f"All outputs saved to '{OUTPUT_DIR}/'")