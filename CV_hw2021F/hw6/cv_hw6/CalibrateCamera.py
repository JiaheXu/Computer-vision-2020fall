import cv2
import os
import numpy as np


def CalibrateCamera( path, width, height, size ):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((width * height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    dirs = os.listdir(path)
    for img_path in dirs:
        img = cv2.imread(os.path.join(path, img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corner_size = (width, height)
        ret, corners = cv2.findChessboardCorners(gray, corner_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    RMS = mean_error / len(objpoints)

    return RMS, K, dist
