import cv2
import os

from canny_detector import CannyEdgeDetector

# --------------------------------------------------
# CREATE OUTPUT FOLDER
# --------------------------------------------------

os.makedirs("Edge_Output", exist_ok=True)

# --------------------------------------------------
# LOAD HL SUBBANDS
# --------------------------------------------------

HL1 = cv2.imread(
    "DWT_Output/HL1.png",
    cv2.IMREAD_GRAYSCALE
)

HL2 = cv2.imread(
    "DWT_Output/HL2.png",
    cv2.IMREAD_GRAYSCALE
)

HL3 = cv2.imread(
    "DWT_Output/HL3.png",
    cv2.IMREAD_GRAYSCALE
)

if HL1 is None or HL2 is None or HL3 is None:
    raise FileNotFoundError("HL images not found.")

print("HL subbands loaded.")

print("HL1 Shape:", HL1.shape)
print("HL2 Shape:", HL2.shape)
print("HL3 Shape:", HL3.shape)

# --------------------------------------------------
# INITIALIZE DETECTOR
# --------------------------------------------------

detector = CannyEdgeDetector()

# --------------------------------------------------
# APPLY CUSTOM CANNY
# --------------------------------------------------

E1 = detector.detect(HL1, 50, 150)
E2 = detector.detect(HL2, 50, 150)
E3 = detector.detect(HL3, 50, 150)

print("Custom Canny completed.")

# --------------------------------------------------
# SAVE EDGE MAPS
# --------------------------------------------------

cv2.imwrite("Edge_Output/E1.png", E1)
cv2.imwrite("Edge_Output/E2.png", E2)
cv2.imwrite("Edge_Output/E3.png", E3)

print("Edge maps saved.")