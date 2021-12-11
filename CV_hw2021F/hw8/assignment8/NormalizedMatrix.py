import cv2
import numpy as np

def NormalizedMatrix( P, K ):
    K_inv = np.linalg.inv(K)

    return np.float32(K_inv.dot(P))