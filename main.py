import os
import sys
import glob
import argparse
import subprocess
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from position_estimation.camera_functions.get_camera_parameters import get_camera_parameters
from position_estimation.players_detection import detect_players
from position_estimation.apply_homography import apply_homography
from temporal_association.connect_players import connect_players

REPO = os.path.dirname(os.path.abspath(__file__))
COURT_SCRIPT = os.path.join(REPO, "padel-court-detector", "detect_court.py")
VIDEO_EXT = (".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".webm")


def parse_args():
    #définition des arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-video_path", "--video_path", required=True, nargs="+",
                    help="vidéo(s), dossier(s) ou glob(s) - mettre le glob entre guillemets pour les globs récursifs (**)")
    ap.add_argument("-frame_start", "--frame_start", type=int, default=0)
    ap.add_argument("-frame_end", "--frame_end", type=int, default=None)
    ap.add_argument("-frame_step", "--frame_step", type=int, default=1)
    ap.add_argument("--out_dir", default="outputs", help="résultats écrits dans <out_dir>/<nom de la vidéo>/")
    ap.add_argument("--workers", type=int, default=1, help="nombre de vidéos traitées en parallèle (sur le même GPU)")
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=True, help="inférence FP16 (--no-half pour FP32)")
    ap.add_argument("--pose_imgsz", type=int, default=960, help="taille d'entrée du modèle pose sur les crops")
    ap.add_argument("--box_model", default="yolo26m.pt", help="poids du détecteur (.pt ou .engine)")
    ap.add_argument("--pose_model", default="yolo26m-pose.pt", help="poids du modèle pose (.pt ou .engine)")
    ap.add_argument("--force_court", action="store_true", help="refaire la détection du court même si calibration.json existe")
    return ap.parse_args()


def collect_videos(items):
    #chaque entrée peut être un fichier, un dossier (non récursif) ou un glob
    videos = []
    for item in items:
        if os.path.isdir(item):
            found = [os.path.join(item, n) for n in sorted(os.listdir(item))]
        elif os.path.isfile(item):
            found = [item]
        else:
            found = sorted(glob.glob(item, recursive=True))
        found = [os.path.abspath(p) for p in found if os.path.isfile(p) and p.lower().endswith(VIDEO_EXT)]
        if not found: sys.exit(f"aucune vidéo trouvée pour : {item}")
        videos += [p for p in found if p not in videos]

    #un dossier de sortie par vidéo, nommé d'après la vidéo : deux vidéos ne peuvent pas avoir le même nom
    stems = {}
    for v in videos:
        stem = os.path.splitext(os.path.basename(v))[0]
        if stem in stems: sys.exit(f"deux vidéos portent le même nom ({stem}) :\n  {stems[stem]}\n  {v}")
        stems[stem] = v
    return videos


def detect_court(video, out, force):
    calibration = os.path.join(out, "calibration.json")
    if os.path.exists(calibration) and not force:
        print(f"[{os.path.basename(video)}] court : calibration.json existante réutilisée")
        return calibration
    if os.path.exists(calibration): os.remove(calibration)

    #detect_court.py écrit ./calibration.json : on le lance depuis le dossier de sortie de la vidéo
    with open(os.path.join(out, "court.log"), "w") as log:
        rc = subprocess.run([sys.executable, COURT_SCRIPT, video], cwd=out, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc == 2 or not os.path.exists(calibration):
        raise RuntimeError(f"détection du court impossible, voir {os.path.join(out, 'court.log')}")
    if rc == 1:
        print(f"[{os.path.basename(video)}] ATTENTION : ajustement du court REJETÉ, calibration utilisée quand même "
              f"(voir court.log, relancer avec --force_court après correction)")
    return calibration


def process_video(video, args):
    out = os.path.join(os.path.abspath(args.out_dir), os.path.splitext(os.path.basename(video))[0])
    os.makedirs(out, exist_ok=True)

    #détection du court (une seule fois par vidéo)
    calibration = detect_court(video, out, args.force_court)

    #détection des joueurs
    model, k, f, cx, cy, H = get_camera_parameters(calibration)
    detect_players(video, args.frame_start, args.frame_end, args.frame_step, model, k, f, cx, cy, H, out_dir=out,
                   half=args.half, pose_imgsz=args.pose_imgsz, box_model=args.box_model, pose_model=args.pose_model)

    #appliquer l'homographie
    apply_homography(H, out)

    #connecter les joueurs
    #connect_players(os.path.join(out, "data_homography.json"))
    return out


def _run_one(video, args):
    #exécuté dans un worker : on renvoie l'erreur plutôt que de faire tomber les autres vidéos
    try:
        return video, process_video(video, args), None
    except Exception:
        return video, None, traceback.format_exc()


def prefetch_models(args):
    #télécharger les poids (détecteur, pose, ReID) et installer onnxruntime une seule fois,
    #avant que plusieurs workers ne tentent de le faire en même temps
    import numpy as np
    import torch
    from ultralytics import YOLO
    from position_estimation.players_detection import BOX_IMGSZ, TRACKER_CFG
    YOLO(args.pose_model, task="pose")
    YOLO(args.box_model, task="detect").track(np.zeros((BOX_IMGSZ, BOX_IMGSZ, 3), np.uint8), imgsz=BOX_IMGSZ,
                                              tracker=TRACKER_CFG, verbose=False, half=args.half)
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def main():
    args = parse_args()
    #FP16 sur CPU est extrêmement lent (et inutile) : on ne le garde que si un GPU CUDA est disponible
    import torch
    if args.half and not torch.cuda.is_available():
        print("pas de GPU CUDA : FP16 désactivé")
        args.half = False
    videos = collect_videos(args.video_path)
    workers = max(1, min(args.workers, len(videos)))
    print(f"{len(videos)} vidéo(s), {workers} worker(s), FP16={'oui' if args.half else 'non'}, sorties dans {os.path.abspath(args.out_dir)}")

    results = []
    if workers == 1:
        results = [_run_one(v, args) for v in videos]
    else:
        prefetch_models(args)
        #spawn obligatoire : CUDA ne supporte pas fork
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(_run_one, v, args) for v in videos]
            for fut in as_completed(futures):
                results.append(fut.result())
                video, out, err = results[-1]
                print(f"[{os.path.basename(video)}] {'terminé -> ' + out if err is None else 'ÉCHEC'}")

    #récapitulatif
    failed = [(v, err) for v, _, err in results if err is not None]
    for v, err in failed:
        print(f"\n=== ÉCHEC : {v}\n{err}", file=sys.stderr)
    print(f"\n{len(results) - len(failed)}/{len(results)} vidéo(s) traitée(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
