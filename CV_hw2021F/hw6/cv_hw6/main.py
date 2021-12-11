import CalibrateCamera
import DisplayKeypointsandLines
import EpipolarLines
import ExtractKeypoints
import FundamentalMatrix
import RectifyImage
import os
import cv2

def main():
    path_calibration = "calibration_images"
    path_test = "test_images"
    width = 12
    height = 9
    size = 0.02
    RMS, K, dist = CalibrateCamera.CalibrateCamera(path_calibration, width, height, size)
    print("RMS:", RMS)

    dirs = os.listdir(path_test)
    img = []
    for fname in dirs:
        img.append(cv2.imread(os.path.join(path_test, fname)))
        newimg = RectifyImage.Rectify(img[len(img)-1], K, dist)
        cv2.imshow("Rectified Image", newimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img1 = img[0]
    img2 = img[1]
    N = 10
    x, xp = ExtractKeypoints.ExtractKeypoints(img1, img2, N)
    F = FundamentalMatrix.FundamentalMatrix(x, xp)
    print("F:", F)
    lines1, lines2 = EpipolarLines.EpipolarLines(x, xp, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = lines2.reshape(-1, 3)
    DisplayKeypointsandLines.DisplayKeypointsandLines(img1, img2, x, xp, lines1, lines2)

if __name__ == "__main__":
    main()
