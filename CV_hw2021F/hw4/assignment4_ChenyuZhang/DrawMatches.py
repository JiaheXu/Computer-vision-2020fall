import cv2


def drawmatches(img1, img2, keypoints1, keypoints2, matches):
    # sort good matches
    good = []
    for match in matches:
        if match[0].distance < 0.5 * match[1].distance:
            good.append([match[0]])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img_match = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good, None, flags=2)

    # Show the matched image
    cv2.imshow("Matches", img_match)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
