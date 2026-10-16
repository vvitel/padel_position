import argparse
import subprocess
from position_estimation.camera_functions.get_camera_parameters import get_camera_parameters
from position_estimation.players_detection import detect_players
from position_estimation.apply_homography import apply_homography
from temporal_association.connect_players import connect_players
from vizualisation.plot_data import plot_data

#définition des arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video_path", "--video_path", required=True, type=str)
ap.add_argument("-frame_start", "--frame_start", required=True, type=int)
ap.add_argument("-frame_end", "--frame_end", required=False, type=int)
ap.add_argument("-frame_step", "--frame_step", required=True, type=int)
args = ap.parse_args()

video_path = args.video_path
start, end = args.frame_start, args.frame_end
step = args.frame_step

#détection du court
subprocess.run(f"python ./padel-court-detector/detect_court.py {video_path}", shell=True)

#détection des joueurs
model, k, f, cx, cy, H = get_camera_parameters()
detect_players(video_path, start, end, step, model, k, f, cx, cy)

#appliquer l'homographie
apply_homography(H)

#association temporelle
array_final, seq_keep = connect_players()

#visualisation
plot_data(array_final, seq_keep)
