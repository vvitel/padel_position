import json
import numpy as np

def get_camera_parameters(path="calibration.json"):
    with open(path, "r") as fp:
        cam = json.load(fp)

    model = cam["lens"]["model"]
    k = cam["lens"]["k"]
    f = cam["lens"]["f"]
    cx = cam["lens"]["cx"]
    cy = cam["lens"]["cy"]

    #undistort_points ne gère que le modèle fisheye à 2 coefficients
    assert model == "fisheye" and len(k) == 2, f"modèle d'objectif non supporté : {model}, k={k}"

    H = np.array(
        cam["homography"],
        dtype=np.float64
    )

    return model, k, f, cx, cy, H
