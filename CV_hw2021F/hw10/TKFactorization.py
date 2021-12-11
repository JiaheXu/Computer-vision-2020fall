import cv2
import numpy as np


def TKFactorization(W):
    u, s, vh = np.linalg.svd(W)
    # v = np.transpose(vh)
    # print("v: ", v)
    # print("v shape: ", vh.shape)
    # print("v[:, 0] shape: ", vh[:, 0].shape)
    M = np.vstack((s[0]*u[:, 0], s[1]*u[:, 1], s[2]*u[:, 2]))
    M = np.transpose(M)
    X = vh[0:3, :]
    print("X shape: ", X.shape)
    # X = np.transpose(X)
    # print("X: ", X)

    return M, X