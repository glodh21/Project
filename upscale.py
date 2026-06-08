import cv2
import numpy as np
import argparse
import os

def calculate_scale_consistency(fine_edge_path, coarse_edge_path):
    fine_edges   = cv2.imread(fine_edge_path,   cv2.IMREAD_GRAYSCALE)
    coarse_edges = cv2.imread(coarse_edge_path, cv2.IMREAD_GRAYSCALE)

    if fine_edges is None or coarse_edges is None:
        print("Error: Could not read edge images")
        return None

    h_fine,   w_fine   = fine_edges.shape
    h_coarse, w_coarse = coarse_edges.shape

    print(f"Fine   edge map  : {fine_edges.shape}   ({fine_edge_path})")
    print(f"Coarse edge map  : {coarse_edges.shape}  ({coarse_edge_path})")
    print(f"Comparing at coarse native resolution: ({h_coarse}, {w_coarse})")

    # Downsample fine to coarse resolution — never upsample coarse
    if (h_fine, w_fine) != (h_coarse, w_coarse):
        fine_downsampled = cv2.resize(
            fine_edges, (w_coarse, h_coarse), interpolation=cv2.INTER_AREA
        )
    else:
        fine_downsampled = fine_edges.copy()

    fine_binary   = (fine_downsampled > 127).astype(np.uint8)
    coarse_binary = (coarse_edges     > 127).astype(np.uint8)

    total_coarse  = np.sum(coarse_binary)
    coarse_in_fine = np.sum((coarse_binary == 1) & (fine_binary == 1))

    consistency = (coarse_in_fine / total_coarse * 100) if total_coarse > 0 else 0.0
    precision   = (coarse_in_fine / (np.sum(fine_binary) + 1e-8)) * 100

    print(f"\nTotal coarse edges            : {total_coarse}")
    print(f"Coarse edges found in fine    : {coarse_in_fine}")
    print(f"Coarse edges NOT in fine      : {total_coarse - coarse_in_fine}")
    print(f"\nConsistency (coarse ⊆ fine)   : {consistency:.2f}%")
    print(f"Precision  (fine ⊇ coarse)    : {precision:.2f}%")

    return consistency, fine_binary, coarse_binary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate scale consistency between two DWT edge maps'
    )
    parser.add_argument('--fine',   '-f', type=str, required=True,
                        help='Path to finer level edges  (e.g. Sol/HL_L1_edges.jpg)')
    parser.add_argument('--coarse', '-c', type=str, required=True,
                        help='Path to coarser level edges (e.g. Sol/HL_L2_edges.jpg  — native, NOT upsampled)')
    args = parser.parse_args()

    os.makedirs('Outputs', exist_ok=True)

    result = calculate_scale_consistency(args.fine, args.coarse)
    if result is None:
        exit(1)