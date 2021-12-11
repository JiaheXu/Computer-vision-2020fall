import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def TKDisplay3D( X ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(X.shape[1]):
        ax.scatter(-X[0, i], -X[1, i], -X[2, i])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()