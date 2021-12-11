
import cv2
import numpy as np
import DrawPolygon


def ProjectiveRectification(pp):
    # Calculate the lines (using cross products)
    pt1 = pp[:, 0]
    pt2 = pp[:, 1]
    pt3 = pp[:, 2]
    pt4 = pp[:, 3]
    line1 = np.cross(pt1, pt2)
    line2 = np.cross(pt2, pt3)
    line3 = np.cross(pt3, pt4)
    line4 = np.cross(pt4, pt1)

    # Calculate points at infinity
    pt_infinity1 = np.cross(line1, line3)
    pt_infinity2 = np.cross(line2, line4)

    # Calculate line at infinity
    line_infinity = np.cross(pt_infinity1, pt_infinity2)
    # print('line_infinity', line_infinity)
    line_infinity[0] = line_infinity[0] / line_infinity[2]
    line_infinity[1] = line_infinity[1] / line_infinity[2]

    # Calculate Hp
    Hp = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [line_infinity[0], line_infinity[1], 1.0]])
    # print('Hp', Hp)
    rect_pts = np.zeros((3, 4))
    rect_pts[:, 0] = np.matmul(Hp, pt1)
    rect_pts[:, 1] = np.matmul(Hp, pt2)
    rect_pts[:, 2] = np.matmul(Hp, pt3)
    rect_pts[:, 3] = np.matmul(Hp, pt4)
    # print('\nPoints after rectification:\n', rect_pts)

    # Calculate the rectified coordinates
    rect_pts_homo = np.copy(np.absolute(rect_pts))
    for i in range(4):
        rect_pts_homo[0, i] = rect_pts_homo[0, i] / rect_pts_homo[2, i]
        rect_pts_homo[1, i] = rect_pts_homo[1, i] / rect_pts_homo[2, i]
        rect_pts_homo[2, i] = 1.0
    print('\nPoints after rectification in homogeneous coordinates:\n', rect_pts_homo)

    # Display the rectified coordinate (using DrawPolygon)
    DrawPolygon.DrawPolygon(rect_pts_homo)

    # Return the homography (3x3 numpy.ndarray)
    return Hp

