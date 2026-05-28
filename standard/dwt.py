
import cv2
import pywt
import numpy as np
import os

# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs("DWT_Output", exist_ok=True)

# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

image = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("lena.png not found.")

print("Original Shape:", image.shape)

# --------------------------------------------------
# NORMALIZATION FUNCTION
# --------------------------------------------------

def normalize(img):
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(norm)

# --------------------------------------------------
# LEVEL 1 DWT
# --------------------------------------------------

LL1, (LH1, HL1, HH1) = pywt.dwt2(image, 'haar')

# Save Level 1
cv2.imwrite("DWT_Output/LL1.png", normalize(LL1))
cv2.imwrite("DWT_Output/LH1.png", normalize(LH1))
cv2.imwrite("DWT_Output/HL1.png", normalize(HL1))
cv2.imwrite("DWT_Output/HH1.png", normalize(HH1))

print("Level 1 saved.")

# --------------------------------------------------
# LEVEL 2 DWT
# Apply on LL1
# --------------------------------------------------

LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar')

# Save Level 2
cv2.imwrite("DWT_Output/LL2.png", normalize(LL2))
cv2.imwrite("DWT_Output/LH2.png", normalize(LH2))
cv2.imwrite("DWT_Output/HL2.png", normalize(HL2))
cv2.imwrite("DWT_Output/HH2.png", normalize(HH2))

print("Level 2 saved.")

# --------------------------------------------------
# LEVEL 3 DWT
# Apply on LL2
# --------------------------------------------------

LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')

# Save Level 3
cv2.imwrite("DWT_Output/LL3.png", normalize(LL3))
cv2.imwrite("DWT_Output/LH3.png", normalize(LH3))
cv2.imwrite("DWT_Output/HL3.png", normalize(HL3))
cv2.imwrite("DWT_Output/HH3.png", normalize(HH3))

print("Level 3 saved.")

print("All DWT subbands stored successfully.")
