import cv2
import numpy as np


def hough(img):
    lines = cv2.HoughLines(img, 1, np.pi/180, 150)
    return lines
