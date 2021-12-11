
import cv2


def canny():
    # return: An image (numpy ndarray) with Canny’s edges
    img = cv2.imread('bt.000.png')
    edges = cv2.Canny(img, 100, 200)
    return edges

