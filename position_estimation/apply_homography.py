import cv2
import json
import numpy as np

def apply_homography(H):
    H = np.linalg.inv(H)

    #ouvrir les données
    with open("data_pose.json", "r") as f:
        dico = json.load(f)

    #transformer les points
    dico_match = {}
    for frame in dico.keys():
        dico_frame, cpt = {}, 0
        for person in range(len(dico[frame])):
            x_img = dico[frame][str(person)][0]
            y_img = dico[frame][str(person)][1]
            point = np.array([[[x_img, y_img]]], dtype=np.float32)

            #appliquer l'homographie
            transform_point = cv2.perspectiveTransform(point, H)
            x_world = transform_point[0][0][0]
            y_world = transform_point[0][0][1]

            #retirer points hors du terrain
            if not (0 <= x_world <= 10 and 0 <= y_world <= 20):
                    continue

            #enregistrer la position
            dico_frame[cpt] = (float(x_world), float(y_world))
            cpt += 1

        dico_match[frame] = dico_frame

    #enregistrement
    with open("data_homography.json", "w") as f:
        json.dump(dico_match, f)

          







