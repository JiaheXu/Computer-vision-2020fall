import cv2
from matplotlib import pyplot as plt
import numpy as np


def DisplayKeypointsandLines( img1, img2, x, xp, lines1, lines2 ):

    def drawlines(img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epipolarlines for the points in img2
            lines - corresponding epipolarlines '''
        r = img1.shape[0]
        c = img1.shape[1]

        for r, pt1, pt2 in zip(lines, pts1, pts2):

            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    img5, img6 = drawlines(img1, img2, lines1, x, xp)
    img3, img4 = drawlines(img2, img1, lines2, xp, x)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()
