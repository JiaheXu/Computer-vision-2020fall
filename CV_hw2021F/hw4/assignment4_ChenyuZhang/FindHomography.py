import cv2
import numpy as np


def findhomography(keypoints1, keypoints2, matches):
    good = []
    for match in matches:
        if match[0].distance < 0.5 * match[1].distance:
            good.append([match[0]])

    src_pts = np.float32([keypoints1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, status = cv2.findHomography(src_pts, dst_pts)

    return homography
