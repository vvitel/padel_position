# padel_position

## Lancer

```bash
# une vidéo
python -B main.py --video_path ./video10.mp4 --frame_start 0 --frame_end 17999 --frame_step 1

# un dossier, ou un glob (entre guillemets), plusieurs vidéos en parallèle sur le même GPU
python -B main.py --video_path ./videos --workers 2
python -B main.py --video_path "./videos/**/*.mp4" --workers 2
```

Résultats dans `outputs/<nom de la vidéo>/` (`--out_dir` pour changer) : `calibration.json`, `court.log`,
`data_pose.json`, `data_homography.json`.

| option | défaut | rôle |
|---|---|---|
| `--frame_start` / `--frame_end` / `--frame_step` | 0 / fin / 1 | appliqués à chaque vidéo |
| `--workers` | 1 | vidéos traitées en parallèle |
| `--half` / `--no-half` | FP16 | inférence FP16 (ignoré sur CPU) |
| `--pose_imgsz` | 960 | taille d'entrée du modèle pose sur les crops (essayer 416) |
| `--box_model` / `--pose_model` | `yolo26m.pt` / `yolo26m-pose.pt` | `.pt` ou `.engine` TensorRT |
| `--force_court` | non | refaire la détection du court même si `calibration.json` existe |

La détection du court n'est faite qu'une fois par vidéo : `calibration.json` est réutilisé aux lancements suivants.
Elle a besoin de `ffmpeg` dans le PATH.

## TensorRT FP16 (V100S)

Le support des GPU Volta (V100 / V100S) a été retiré de TensorRT 10.5 : installer TensorRT 10.4 **avant**
l'export, sinon Ultralytics installe la dernière version. PyTorch doit être un build CUDA 12.

```bash
pip install "tensorrt-cu12==10.4.0"
yolo export model=yolo26m.pt      format=engine half=True imgsz=960 batch=1 device=0
yolo export model=yolo26m-pose.pt format=engine half=True imgsz=416 dynamic=True batch=8 device=0

python -B main.py --video_path ./videos --box_model yolo26m.engine --pose_model yolo26m-pose.engine --pose_imgsz 416
```

Un `.engine` n'est valable que pour le GPU et la version de TensorRT qui l'ont construit.
`--pose_imgsz` doit être égal à l'`imgsz` de l'export du modèle pose.
