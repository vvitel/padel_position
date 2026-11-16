import os
import cv2
import json
import numpy as np

#dimensions du terrain (m) : x de 0 à 10, y de 0 à 20
COURT_WIDTH = 10
COURT_LENGTH = 20


def image_to_world(H_inv, x_img, y_img):
    #pixel non distordu -> mètres sur le terrain (H_inv = inverse de l'homographie terrain -> image)
    point = np.array([[[x_img, y_img]]], dtype=np.float32)
    transform_point = cv2.perspectiveTransform(point, H_inv)
    return float(transform_point[0][0][0]), float(transform_point[0][0][1])


def in_court(x_world, y_world):
    return 0 <= x_world <= COURT_WIDTH and 0 <= y_world <= COURT_LENGTH


def apply_homography(H, out_dir="."):
    H = np.linalg.inv(H)

    #ouvrir les données
    with open(os.path.join(out_dir, "data_pose.json"), "r") as fp:
        dico = json.load(fp)

    #transformer les points
    dico_match = {}
    for frame in dico.keys():
        lst_frame = []
        for person in dico[frame].keys():
            track_id = dico[frame][person]["id"]
            x_img, y_img = dico[frame][person]["position"]

            #appliquer l'homographie
            x_world, y_world = image_to_world(H, x_img, y_img)

            #retirer points hors du terrain
            if not in_court(x_world, y_world):
                continue

            #enregistrer les données
            lst_frame.append((track_id, x_world, y_world))

        dico_match[frame] = lst_frame

    #enregistrement
    with open(os.path.join(out_dir, "data_homography.json"), "w") as fp:
        json.dump(dico_match, fp)
