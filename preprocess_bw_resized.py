## Import
import numpy as np
import cv2

def proc_to_bw_resized(img : np.ndarray )-> np.ndarray :
    """
    Process an image from cv2.imread(img_path) : convert to black and white
    and resize to (64,64)
    """
    img_cv2 = img[...,::-1] # Convert BGR to RGB
    res = cv2.resize(img_cv2, dsize=(64, 64))     # Resizing
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) # Convert to Grayscale
    return gray_image
