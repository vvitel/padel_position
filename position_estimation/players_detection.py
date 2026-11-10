import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO
from position_estimation.camera_functions.undistort_points import undistort_points
from position_estimation.pose_functions.find_foot_with_nose import camera_pose_from_homography, find_foot_with_nose

#hauteur du nez (m) pour un joueur de 1,75 m
NOSE_HEIGHT = 1.63
#confiance minimale pour considérer un keypoint comme visible
KEYPOINT_CONF = 0.5

def detect_players(video_path, start, end, step, model, k, f, cx, cy, H):
    #chargement du modèles
    model_box = YOLO("yolo26m.pt")
    model_pose = YOLO("yolo26m-pose.pt")

    #paramètres caméra
    cam = {"model": model, "k": k, "f": f, "cx": cx, "cy": cy}
    R, C = camera_pose_from_homography(H, f, cx, cy)

    #lecture de la vidéo
    cap = cv2.VideoCapture(video_path)
    try:
        if end is None: end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #définir la première frame
        frame_index = start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        dico = {}
        #parcourir les images
        with tqdm(total=max(end - start, 0)) as pbar:
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
                            kp = keypoints[0]

                            #si aucune cheville n'est visible, estimer les pieds à partir du nez
                            #(sinon on garde le centre bas de la box, traité plus bas)
                            if not (kp[15, 2] > KEYPOINT_CONF or kp[16, 2] > KEYPOINT_CONF):
                                #récupérer le point du nez
                                if kp[0, 2] <= KEYPOINT_CONF: continue
                                #repasser des coordonnées du crop à celles de l'image
                                x_nose, y_nose = float(kp[0, 0]) + x1, float(kp[0, 1]) + y1
                                nose_undistorted = undistort_points([[x_nose, y_nose]], cam)
                                x_nose_undistorted, y_nose_undistorted = nose_undistorted[0]
                                #estimer position des pieds (en pixels non distordus)
                                foot = find_foot_with_nose(x_nose_undistorted, y_nose_undistorted, R, C, H, f, cx, cy, NOSE_HEIGHT)
                                if foot is None: continue
                                dic_pos_frame[cpt] = {"id": track_id,
                                                      "position": list(foot)}
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
    finally:
        cap.release()

    #enregistrer les résultats
    with open("data_pose.json", "w") as fp:
        json.dump(dico, fp)
