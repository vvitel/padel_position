import numpy as np
from scipy.optimize import least_squares
from position_estimation.camera_functions.distort import distort

def undistort_points(points_px, cam):
    points_px = np.asarray(points_px, dtype=float)

    xy_distorted = np.column_stack([
        (points_px[:, 0] - cam["cx"]) / cam["f"],
        (points_px[:, 1] - cam["cy"]) / cam["f"]
    ])

    points_ideal = []

    for target in xy_distorted:
        result = least_squares(
            lambda xy: distort(
                xy.reshape(1, 2), cam["k"])[0] - target, target
        )
        points_ideal.append(result.x)

    points_ideal = np.asarray(points_ideal)

    points_undistorted_px = np.column_stack([
        points_ideal[:, 0] * cam["f"] + cam["cx"],
        points_ideal[:, 1] * cam["f"] + cam["cy"]
    ])

    return points_undistorted_px