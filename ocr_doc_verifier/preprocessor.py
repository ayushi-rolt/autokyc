import cv2
import numpy as np

def preprocess_image(gray_image):
  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray_image)

    denoised = cv2.fastNlMeansDenoising(contrast, h=30)

    adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

  

    coords = np.column_stack(np.where(morph > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = morph.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(morph, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed
