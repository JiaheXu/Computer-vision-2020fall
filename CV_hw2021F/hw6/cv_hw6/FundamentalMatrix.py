import cv2

def FundamentalMatrix( x, xp ):
    F, mask = cv2.findFundamentalMat(x, xp, cv2.FM_8POINT)
    return F
