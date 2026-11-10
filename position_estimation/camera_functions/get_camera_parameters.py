import json
import numpy as np

def get_camera_parameters():
    with open("calibration.json", "r") as f:
        cam = json.load(f)

    model = cam["lens"]["model"]
    k = cam["lens"]["k"]
    f = cam["lens"]["f"]
    cx = cam["lens"]["cx"]
    cy = cam["lens"]["cy"]

    H = np.array(
        cam["homography"],
        dtype=np.float64
    )

    return model, k, f, cx, cy, H