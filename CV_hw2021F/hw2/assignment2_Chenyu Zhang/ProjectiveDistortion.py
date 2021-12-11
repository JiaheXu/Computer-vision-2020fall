
import cv2
import numpy as np


def ProjectiveDistortion(ph):
    # print(ph)
    H = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [1, 1, 1]])
    ph_new = np.matmul(H, ph)
    # print('\nPoints after distortion:\n', ph_new)
    ph_new_homo = np.copy(ph_new)
    for i in range(4):
        ph_new_homo[0, i] = ph_new_homo[0, i] / ph_new_homo[2, i]
        ph_new_homo[1, i] = ph_new_homo[1, i] / ph_new_homo[2, i]
        ph_new_homo[2, i] = 1.0
    print('\nPoints after distortion in homogeneous coordinates:\n', ph_new_homo)
    return ph_new_homo


