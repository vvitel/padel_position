import cv2
import json
import numpy as np

def apply_homography(H):
    H = np.linalg.inv(H)

    #ouvrir les données
    with open("data_pose.json", "r") as fp:
        dico = json.load(fp)

    #transformer les points
    dico_match = {}
    for frame in dico.keys():
        lst_frame = []
        for person in dico[frame].keys():
            track_id = dico[frame][person]["id"]
            x_img, y_img = dico[frame][person]["position"]
            point = np.array([[[x_img, y_img]]], dtype=np.float32)

            #appliquer l'homographie
            transform_point = cv2.perspectiveTransform(point, H)
            x_world = transform_point[0][0][0]
            y_world = transform_point[0][0][1]

            #retirer points hors du terrain
            if not (0 <= x_world <= 10 and 0 <= y_world <= 20):
                continue

            #enregistrer les données
            lst_frame.append((track_id, float(x_world), float(y_world)))

        dico_match[frame] = lst_frame

    #enregistrement
    with open("data_homography.json", "w") as fp:
        json.dump(dico_match, fp)
