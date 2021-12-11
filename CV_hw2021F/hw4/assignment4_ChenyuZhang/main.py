import Canny
import Hough
import DrawLines
import cv2
import SIFT
import Matching
import DrawMatches
import FindHomography
import WarpImage


def main():
    img = cv2.imread('bt.000.png')
    edges = Canny.canny()
    lines = Hough.hough(edges)
    DrawLines.drawlines(img, lines)

    img_box = cv2.imread('box.pgm')
    img_scene = cv2.imread('scene.pgm')
    kp_box, des_box = SIFT.sift('box.pgm')
    kp_scene, des_scene = SIFT.sift('scene.pgm')
    matches = Matching.matching(des_box, des_scene, 7)
    DrawMatches.drawmatches(img_box, img_scene, kp_box, kp_scene, matches)
    h = FindHomography.findhomography(kp_box, kp_scene, matches)
    print('H: ', h)
    warped_img = WarpImage.warpimage(img_box, h)
    cv2.imshow("Warped image", warped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
