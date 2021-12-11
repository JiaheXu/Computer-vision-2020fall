import cv2
import numpy as np
from scipy.linalg import null_space

def CanonicalCameras( F ):
    # e_p = null_space(np.transpose(F))
    # P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    # P_p = np.hstack((np.cross(e_p, F), e_p))
    e_p = null_space(np.transpose(F)).reshape(-1)
    P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    skew_ep = np.asarray([[0, -e_p[2], e_p[1]], [e_p[2], 0, -e_p[0]], [-e_p[1], e_p[0], 0]])
    P_p = np.hstack((skew_ep.dot(F), e_p.reshape(3, 1)))

    return np.float32(P), np.float32(P_p)