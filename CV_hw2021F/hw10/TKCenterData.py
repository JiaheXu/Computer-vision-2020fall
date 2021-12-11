import cv2
import numpy as np


def TKCenterData( W, t ):
    m = W.shape[0]/2
    m = int(m)
    n = W.shape[1]
    n = int(n)
    for i in range(m):
        for j in range(n):
            W[2 * i, j] = W[2 * i, j] - t[0, i]
            W[2 * i + 1, j] = W[2 * i + 1, j] - t[1, i]

    return W
