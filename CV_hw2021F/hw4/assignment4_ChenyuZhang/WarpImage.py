import cv2
import numpy as np


def warpimage(img, H):
    warped_image = cv2.warpPerspective(img, H, (512, 384))
    return warped_image
