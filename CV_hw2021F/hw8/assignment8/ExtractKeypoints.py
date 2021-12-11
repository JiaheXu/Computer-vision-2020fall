import cv2
import numpy as np


def ExtractKeypoints(img1, img2, N):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    good = matches[:N]

    src_pts = [kp1[m.queryIdx] for m in good]
    dst_pts = [kp2[m.trainIdx] for m in good]

    src_pts = cv2.KeyPoint_convert(src_pts)
    dst_pts = cv2.KeyPoint_convert(dst_pts)

    return np.float32(src_pts), np.float32(dst_pts)