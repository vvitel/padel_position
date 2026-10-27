import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO
from position_estimation.camera_functions.undistort_points import undistort_points

def detect_players(video_path, start, end, step, model, k, f, cx, cy):
    #chargement du modèles
    model_box = YOLO("yolo26m.pt")

    #lecture de la vidéo
    cap = cv2.VideoCapture(video_path)
    if end is None: end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #paramètres caméra
    cam = {"model": model, "k": k, "f": f, "cx": cx, "cy": cy}

    #définir la première frame
    frame_index = start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    dico = {}
    #parcourir les images
    with tqdm() as pbar:
        while True:
            ret, frame = cap.read()

            if not ret or frame_index >= end: break

            #appliquer le modèle box
            if frame_index % step == 0:
                result_box = model_box.track(frame, imgsz=960, classes=[0], tracker="config_tracking.yaml", verbose=False, persist=True,)
                boxes = result_box[0].boxes

                #parcourir les détections
                dic_pos_frame, cpt = {}, 0
                for box in boxes:
                    #récupérer le point centre bas de la box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bottom_center_x = (x1 + x2) / 2
                    bottom_center_y = y2

                    #récupérer l'id de tracking
                    track_id = int(box.id[0]) if box.id is not None else None

                    #si la box atteint le bas de l'image
                    if bottom_center_y >= frame.shape[0] - 2:
                        height = (x2 - x1) / 0.4
                        bottom_center_y = y1 + height

                    #retirer la distortion
                    point_undistorted = undistort_points(
                        [[bottom_center_x, bottom_center_y]],
                        cam
                    )

                    #garder la position
                    dic_pos_frame[cpt] = {"id": track_id,
                                          "position": point_undistorted[0].tolist()}
                    cpt += 1

                dico[frame_index] = dic_pos_frame 
                
            frame_index += 1
            pbar.update(1)

    #enregistrer les résultats
    with open("data_pose.json", "w") as f:
        json.dump(dico, f)

    cap.release()