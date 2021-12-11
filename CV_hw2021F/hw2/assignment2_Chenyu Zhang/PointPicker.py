import cv2
import numpy as np


def PointPicker():
    N = 4
    print('Please select four points in clockwise or counterclockwise'
          ' by clicking left button of the mouse. \nPress ESC to close the image.')
    Pts = []

    def pick_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            Pts.append((x, y))
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

    # Read the square image from a file
    img = cv2.imread('dots.jpg', 0)
    cv2.namedWindow('Dots Image')

    # Enable the selection of points in the image with left mouse clicks
    cv2.setMouseCallback('Dots Image', pick_points)

    while True:
        # Display the image in a window
        cv2.imshow('Dots Image', img)
        k = cv2.waitKey(1)
        if k == 27:   # Close the window
            break

    cv2.destroyAllWindows()
    if len(Pts) < N:
        print('ERROR: Please select no less than four points!')
        return
    print('\nThe coordinates of the selected points:\n', Pts)

    # Return the normalized homogeneous coordinate of the selected coordinates (last row of 1s)
    homo_coors = np.zeros((3, N))
    for i in range(N):
        pt_in_list = list(Pts[i])
        pt_x = float(pt_in_list[0])/300.0
        pt_y = float(pt_in_list[1])/300.0
        homo_coors[0, i] = pt_x
        homo_coors[1, i] = pt_y
        homo_coors[2, i] = 1.0
    print('\nThe homogeneous coordinates of the selected points:\n', homo_coors)
    return homo_coors




