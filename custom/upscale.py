import cv2
import numpy as np
import os

# --------------------------------------------------
# CREATE OUTPUT FOLDERS
# --------------------------------------------------

os.makedirs("Containment_Output", exist_ok=True)
os.makedirs("Composite_Output", exist_ok=True)

# --------------------------------------------------
# LOAD EDGE MAPS
# --------------------------------------------------

E1 = cv2.imread(
    "Edge_Output/E1.png",
    cv2.IMREAD_GRAYSCALE
)

E2 = cv2.imread(
    "Edge_Output/E2.png",
    cv2.IMREAD_GRAYSCALE
)

E3 = cv2.imread(
    "Edge_Output/E3.png",
    cv2.IMREAD_GRAYSCALE
)

if E1 is None or E2 is None or E3 is None:
    raise FileNotFoundError("Edge maps not found.")

print("Edge maps loaded.")

print("E1 Shape:", E1.shape)
print("E2 Shape:", E2.shape)
print("E3 Shape:", E3.shape)

# --------------------------------------------------
# UPSAMPLE TO MATCH E1
# --------------------------------------------------

target_h, target_w = E1.shape

E2_up = cv2.resize(
    E2,
    (target_w, target_h),
    interpolation=cv2.INTER_NEAREST
)

E3_up = cv2.resize(
    E3,
    (target_w, target_h),
    interpolation=cv2.INTER_NEAREST
)

cv2.imwrite(
    "Containment_Output/E2_up.png",
    E2_up
)

cv2.imwrite(
    "Containment_Output/E3_up.png",
    E3_up
)

print("Upsampled maps saved.")

# --------------------------------------------------
# BINARY CONVERSION
# --------------------------------------------------

E1_bin = (E1 > 0).astype(np.uint8)
E2_bin = (E2_up > 0).astype(np.uint8)
E3_bin = (E3_up > 0).astype(np.uint8)

# --------------------------------------------------
# CONTAINMENT
# --------------------------------------------------

intersection_32 = np.logical_and(
    E3_bin,
    E2_bin
)

containment_32 = (
    np.sum(intersection_32)
    /
    (np.sum(E3_bin) + 1e-8)
) * 100

intersection_21 = np.logical_and(
    E2_bin,
    E1_bin
)

containment_21 = (
    np.sum(intersection_21)
    /
    (np.sum(E2_bin) + 1e-8)
) * 100

print("\n===== RESULTS =====")

print(
    f"E3 -> E2 Containment: "
    f"{containment_32:.2f}%"
)

print(
    f"E2 -> E1 Containment: "
    f"{containment_21:.2f}%"
)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

with open(
    "Containment_Output/results.txt",
    "w"
) as f:

    f.write(
        f"E3 -> E2: "
        f"{containment_32:.2f}%\n"
    )

    f.write(
        f"E2 -> E1: "
        f"{containment_21:.2f}%\n"
    )

print("Results saved.")
# --------------------------------------------------
# CREATE COMPOSITE OVERLAP IMAGES
# --------------------------------------------------

# --------------------------------------------------
# Composite 1:
# E3_up vs E2_up
# --------------------------------------------------

comp_32 = np.zeros(
    (target_h, target_w, 3),
    dtype=np.uint8
)

# Green = E3 only
comp_32[
    (E3_bin == 1) & (E2_bin == 0)
] = [0, 255, 0]

# Red = E2 only
comp_32[
    (E3_bin == 0) & (E2_bin == 1)
] = [255, 0, 0]

# Yellow = overlap
comp_32[
    (E3_bin == 1) & (E2_bin == 1)
] = [255, 255, 0]

cv2.imwrite(
    "Composite_Output/E3_vs_E2_overlap.png",
    cv2.cvtColor(comp_32, cv2.COLOR_RGB2BGR)
)

# --------------------------------------------------
# Composite 2:
# E2_up vs E1
# --------------------------------------------------

comp_21 = np.zeros(
    (target_h, target_w, 3),
    dtype=np.uint8
)

# Green = E2 only
comp_21[
    (E2_bin == 1) & (E1_bin == 0)
] = [0, 255, 0]

# Red = E1 only
comp_21[
    (E2_bin == 0) & (E1_bin == 1)
] = [255, 0, 0]

# Yellow = overlap
comp_21[
    (E2_bin == 1) & (E1_bin == 1)
] = [255, 255, 0]

cv2.imwrite(
    "Composite_Output/E2_vs_E1_overlap.png",
    cv2.cvtColor(comp_21, cv2.COLOR_RGB2BGR)
)

print("Composite overlap images saved.")