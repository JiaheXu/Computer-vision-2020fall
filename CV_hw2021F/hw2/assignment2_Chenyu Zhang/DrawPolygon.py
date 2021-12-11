import cv2
import numpy
import numpy as np


def DrawPolygon(p):
    # Create a white image large enough to draw all the coordinates
    img = np.zeros([400, 400, 3])
    img.fill(255)
    # print('Image shape:', img.shape)

    # Draw black lines of the polygon defined by the points
    pts = np.zeros((4, 2))
    for i in range(4):
        x = int(p[0, i]*300)
        y = int(p[1, i]*300)
        pts[i, 0] = x
        pts[i, 1] = y
        cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
    cv2.polylines(img, np.int32([pts]), True, (0, 0, 255), 2)

    # Display the image in a window and close the image manually
    cv2.imshow('Polygon', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

