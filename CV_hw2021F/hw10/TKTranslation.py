import cv2
import numpy as np


def TKTranslation( W ):
    m = W.shape[0]/2
    m = int(m)
    # print(m)
    t = np.zeros((2, m))
    for i in range(m):
        t[0, i] = np.average(W[2 * i, :])
        t[1, i] = np.average(W[2 * i + 1, :])

    return t
