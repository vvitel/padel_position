import os
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
#taille d'entrée du modèle box (doit correspondre à l'export TensorRT si un .engine est utilisé)
BOX_IMGSZ = 960
#configuration du tracker, résolue depuis la racine du repo pour ne pas dépendre du dossier courant
TRACKER_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_tracking.yaml")


def detect_players(video_path, start, end, step, model, k, f, cx, cy, H, out_dir=".",
                   half=True, pose_imgsz=960, box_model="yolo26m.pt", pose_model="yolo26m-pose.pt"):
    #chargement des modèles (rechargés pour chaque vidéo : le tracker ne doit pas garder l'état de la précédente)
    model_box = YOLO(box_model, task="detect")
    model_pose = YOLO(pose_model, task="pose")

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
        with tqdm(total=max(end - start, 0), desc=os.path.basename(video_path)) as pbar:
            while True:
                ret, frame = cap.read()

                if not ret or frame_index >= end: break

                #appliquer le modèle box
                if frame_index % step == 0:
                    result_box = model_box.track(frame, imgsz=BOX_IMGSZ, classes=[0], conf=0.05, iou=0.65, tracker=TRACKER_CFG,
                                                 verbose=False, persist=True, quantize=16 if half else None)
                    dico[frame_index] = _positions_in_frame(frame, result_box[0].boxes, model_pose, pose_imgsz, half,
                                                            cam, R, C, H, f, cx, cy)

                frame_index += 1
                pbar.update(1)
    finally:
        cap.release()

    #enregistrer les résultats
    with open(os.path.join(out_dir, "data_pose.json"), "w") as fp:
        json.dump(dico, fp)


def _positions_in_frame(frame, boxes, model_pose, pose_imgsz, half, cam, R, C, H, f, cx, cy):
    #un seul transfert GPU -> CPU par frame plutôt qu'un par box
    xyxy = boxes.xyxy.cpu().numpy().astype(float)
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

    #boxes qui atteignent le bas de l'image : on regroupe leurs crops pour un seul appel au modèle pose
    crops, crop_of = [], {}
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        if y2 >= frame.shape[0] - 2:
            ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
            crop = frame[iy1:iy2, ix1:ix2]
            if crop.size == 0: continue
            crop_of[i] = len(crops)
            crops.append(crop)
    poses = model_pose.predict(source=crops, imgsz=pose_imgsz, show=False, verbose=False, classes=[0], quantize=16 if half else None) if crops else []

    #parcourir les détections
    dic_pos_frame, cpt = {}, 0
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        track_id = ids[i]
        #récupérer le point centre bas de la box
        bottom_center_x = (x1 + x2) / 2
        bottom_center_y = y2

        #si la box atteint le bas de l'image
        if bottom_center_y >= frame.shape[0] - 2:
            if i not in crop_of: continue
            keypoints = poses[crop_of[i]].keypoints.data
            if len(keypoints) == 0: continue
            kp = keypoints[0].cpu()

            #si aucune cheville n'est visible, estimer les pieds à partir du nez
            #(sinon on garde le centre bas de la box, traité plus bas)
            if not (kp[15, 2] > KEYPOINT_CONF or kp[16, 2] > KEYPOINT_CONF):
                #récupérer le point du nez
                if kp[0, 2] <= KEYPOINT_CONF: continue
                #repasser des coordonnées du crop à celles de l'image
                x_nose, y_nose = float(kp[0, 0]) + int(x1), float(kp[0, 1]) + int(y1)
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
            [[float(bottom_center_x), float(bottom_center_y)]],
            cam
        )

        #garder la position
        dic_pos_frame[cpt] = {"id": track_id,
                              "position": point_undistorted[0].tolist()}
        cpt += 1

    return dic_pos_frame
