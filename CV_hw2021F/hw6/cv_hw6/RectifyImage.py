import cv2
import numpy
import sys
import math
import os

def Rectify( img, K, dist ):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newcameramtx, (w, h), 5)
    newimg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    return newimg
