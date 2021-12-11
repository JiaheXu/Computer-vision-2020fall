import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import null_space


def FundamentalMatrix( x, xp ):
# x and xp are column vectors
    print("x:", x)
    print("xp:", xp)
    x = x[:2, :]
    xp = xp[:2, :]
    x = x.T
    xp = xp.T
    F_hat, mask = cv2.findFundamentalMat(x, xp, cv2.FM_8POINT)
    # print("F by openCV:", F_hat)
    # print("\n")
    e_p = null_space(np.transpose(F_hat)).reshape(-1)
    P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    skew_ep = np.asarray([[0, -e_p[2], e_p[1]], [e_p[2], 0, -e_p[0]], [-e_p[1], e_p[0], 0]])
    P_p = np.hstack((skew_ep.dot(F_hat), e_p.reshape(3, 1)))
    x = np.transpose(x)
    xp = np.transpose(xp)

    points_3D = cv2.triangulatePoints(P, P_p, x, xp)
    # print("point 3D:", points_3D)
    points_3D = np.divide(points_3D[:3, :], points_3D[3, :])
    data0 = np.hstack((P_p[0, :], P_p[1, :], P_p[2, :]))

    for i in range(points_3D.shape[1]):
        data0 = np.hstack((data0, np.transpose(points_3D[:, i])))

    # print("data0: ", data0)

    def cost_func(data, x, xp):
        # x and xp are row vectors
        x = np.transpose(x)
        xp = np.transpose(xp)
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        P_p = np.vstack((data[:4], data[4:8], data[8:12]))
        x_hat = []
        xp_hat = []
        for i in range(int((len(data)-12)/3)):
            point_3D = data[12+3*i:15+3*i]
            point_3D = np.hstack((point_3D, 1))
            x_hat.append(P.dot(point_3D))
            xp_hat.append(P_p.dot(point_3D))
        # x_hat and xp_hat are row vectors
        x_stack = np.hstack((x[0, :], xp[0, :]))
        x_hat_stack = np.hstack((x_hat[0], xp_hat[0]))
        for i in range(1, len(x_hat)):

            x_stack = np.hstack((x_stack, x[i, :], xp[i, :]))
            x_hat_stack = np.hstack((x_hat_stack, x_hat[i], xp_hat[i]))

        cost = x_stack - x_hat_stack
        return cost


    ones = np.asarray([1] * x.shape[1])
    x_homo = np.vstack((x, ones))
    xp_homo = np.vstack((xp, ones))

    result = least_squares(cost_func, data0, args=(x_homo, xp_homo), method='lm')
    data_opt = result.x
    P_p = np.vstack((data_opt[:4], data_opt[4:8], data_opt[8:12]))
    M = P_p[:3, :3]
    t = P_p[:3, 3].reshape(-1)
    skew_t = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = skew_t.dot(M)

    return np.float32(F)







