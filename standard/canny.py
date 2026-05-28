import cv2
import os

# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs("Edge_Output", exist_ok=True)

# --------------------------------------------------
# LOAD SUBBANDS
# --------------------------------------------------

HL1 = cv2.imread("DWT_Output/HL1.png", cv2.IMREAD_GRAYSCALE)
HL2 = cv2.imread("DWT_Output/HL2.png", cv2.IMREAD_GRAYSCALE)
HL3 = cv2.imread("DWT_Output/HL3.png", cv2.IMREAD_GRAYSCALE)

if HL1 is None or HL2 is None or HL3 is None:
    raise FileNotFoundError("HL subband images not found.")

# --------------------------------------------------
# APPLY STANDARD CANNY
# --------------------------------------------------

E1 = cv2.Canny(HL1, 50, 150)
E2 = cv2.Canny(HL2, 50, 150)
E3 = cv2.Canny(HL3, 50, 150)

# --------------------------------------------------
# SAVE ORIGINAL EDGE MAPS
# --------------------------------------------------

cv2.imwrite("Edge_Output/E1_256x256.png", E1)
cv2.imwrite("Edge_Output/E2_128x128.png", E2)
cv2.imwrite("Edge_Output/E3_64x64.png", E3)

print("Original edge maps saved.")

# --------------------------------------------------
# UPSCALE EDGE MAPS
# --------------------------------------------------

E1_up = cv2.resize(E1, (512, 512), interpolation=cv2.INTER_NEAREST)
E2_up = cv2.resize(E2, (512, 512), interpolation=cv2.INTER_NEAREST)
E3_up = cv2.resize(E3, (512, 512), interpolation=cv2.INTER_NEAREST)

# --------------------------------------------------
# SAVE UPSCALED EDGE MAPS
# --------------------------------------------------

cv2.imwrite("Edge_Output/E1_upscaled_512.png", E1_up)
cv2.imwrite("Edge_Output/E2_upscaled_512.png", E2_up)
cv2.imwrite("Edge_Output/E3_upscaled_512.png", E3_up)

print("Upscaled edge maps saved.")

print("All edge maps generated successfully.")