import cv2


def sift(filename):
    # return: keypoints, descriptors

    img = cv2.imread(filename)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    return kp, des
