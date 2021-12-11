
import cv2
import numpy as np

def OpenCVRectification(p, pp):
    p = p[0:2, :]
    pp = pp[0:2, :]
    # print('\np', p)
    # print('pp', pp)
    src_pts = np.transpose(p)
    dst_pts = np.transpose(pp)
    # print('src_pts', src_pts)
    # print('dst_pts', dst_pts)
    h, status = cv2.findHomography(src_pts, dst_pts)
    return h
