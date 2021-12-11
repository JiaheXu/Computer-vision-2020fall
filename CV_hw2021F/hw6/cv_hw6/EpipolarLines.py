import cv2
import numpy as np


def EpipolarLines( x, xp, F ):
    lines1 = cv2.computeCorrespondEpilines(xp.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(3, 1, -1)
    lines2 = cv2.computeCorrespondEpilines(x.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(3, 1, -1)
    print("line shape:", lines1.shape)
    return np.float32(lines1), np.float32(lines2)
