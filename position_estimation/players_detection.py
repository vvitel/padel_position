import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO
from position_estimation.camera_functions.undistort_points import undistort_points
from position_estimation.pose_functions.find_foot_with_nose import find_foot_with_nose

def detect_players(video_path, start, end, step, model, k, f, cx, cy, H):
    #chargement du modèles
    model_box = YOLO("yolo26m.pt")
    model_pose = YOLO("yolo26m-pose.pt")

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
                result_box = model_box.track(frame, imgsz=960, classes=[0], conf=0.15, iou=0.65, tracker="config_tracking.yaml", verbose=False, persist=True)
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
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        crop = frame[y1:y2, x1:x2]
                        result_pose = model_pose.predict(source=crop, imgsz=960, show=False, verbose=False, classes=[0])
                        keypoints = result_pose[0].keypoints.data
                        if len(keypoints) == 0: continue
                        #si on a quand même une cheville
                        left_ankle, right_ankle = keypoints[0][15], keypoints[0][16]
                        x_left_ankle, y_left_ankle = left_ankle[:2]
                        x_right_ankle, y_right_ankle = right_ankle[:2]
                        if (x_left_ankle != 0) or (y_left_ankle != 0) or (x_right_ankle != 0) or (y_right_ankle != 0):
                            point_undistorted = undistort_points([[bottom_center_x, bottom_center_y]], cam)
                            dic_pos_frame[cpt] = {"id": track_id, "position": point_undistorted[0].tolist()}
                            cpt += 1
                            continue
                        else:
                            #récupérer le point du nez
                            nose = keypoints[0][0]
                            x_nose, y_nose = nose[:2]
                            if (x_nose == 0) and (y_nose == 0): continue
                            nose_undistorted = undistort_points([[x_nose, y_nose]], cam)
                            x_nose_undistorted, y_nose_undistorted = nose_undistorted[0]
                            #estimer position des pieds
                            x_foot_estimated, y_foot_estimated = find_foot_with_nose(x_nose_undistorted, y_nose_undistorted, H, f, cx, cy, 1.75)
                            estimated_point = [x_foot_estimated, y_foot_estimated]
                            dic_pos_frame[cpt] = {"id": track_id,
                                                "position": estimated_point}
                            cpt += 1
                            continue

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