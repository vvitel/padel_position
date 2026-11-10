import cv2
import numpy as np

#critère d'arrêt de la résolution itérative d'OpenCV
CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 50, 1e-12)

def undistort_points(points_px, cam):
    points_px = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)

    K = np.array([[cam["f"], 0, cam["cx"]], [0, cam["f"], cam["cy"]], [0, 0, 1]], dtype=np.float64)
    #modèle fisheye OpenCV : theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
    D = np.array([cam["k"][0], cam["k"][1], 0.0, 0.0], dtype=np.float64)

    #P=K pour retrouver des coordonnées en pixels (non distordus)
    points_undistorted_px = cv2.fisheye.undistortPoints(points_px, K, D, P=K, criteria=CRITERIA)

    return points_undistorted_px.reshape(-1, 2)
