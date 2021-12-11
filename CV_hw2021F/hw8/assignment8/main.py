import numpy as np

import CanonicalCameras
import ExtractKeypoints
import FundamentalMatrix
import NormalizedMatrix
import os
import cv2


def main():
    K = np.array([614.5482788085938, 0.0, 319.5,
                  0.0, 614.5482788085938, 239.8736572265625,
                  0.0, 0.0, 1.0])
    K = np.vstack((K[:3], K[3:6], K[6:9]))
    path_test = "test_images"
    dirs = os.listdir(path_test)
    img = []
    for fname in dirs:
        img.append(cv2.imread(os.path.join(path_test, fname)))
    img1 = img[2]
    img2 = img[3]
    N = 10
    x, xp = ExtractKeypoints.ExtractKeypoints(img1, img2, N)
    x = x.T
    xp = xp.T
    F = FundamentalMatrix.FundamentalMatrix(x, xp)
    F = F/F[2, 2]
    print("F:", F)
    print("\n")
    P, Pp = CanonicalCameras.CanonicalCameras(F)
    P_norm = NormalizedMatrix.NormalizedMatrix(P, K)
    Pp_norm = NormalizedMatrix.NormalizedMatrix(Pp, K)
    print("Normalized P: ", P_norm)
    print("\n")
    print("Normalized P':", Pp_norm)

if __name__ == "__main__":
    main()
