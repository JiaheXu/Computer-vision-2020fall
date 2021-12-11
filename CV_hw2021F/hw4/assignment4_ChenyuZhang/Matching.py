import cv2
import numpy as np


def matching(descriptor1, descriptor2, N):
    # return: A list of the N strongest matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=N)
    return matches
