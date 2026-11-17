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
`data_pose.json`, `data_homography.json`, `data_offcourt.json` (points projetés hors du terrain),
`data_tracks.json` (même format que `data_homography.json`, avec les ids réassociés après les disparitions).

| option | défaut | rôle |
|---|---|---|
| `--frame_start` / `--frame_end` / `--frame_step` | 0 / fin / 1 | appliqués à chaque vidéo |
| `--workers` | 1 | vidéos traitées en parallèle |
| `--half` / `--no-half` | FP16 | inférence FP16 (ignoré sur CPU) |
| `--pose_imgsz` | 960 | taille d'entrée du modèle pose sur les crops (essayer 416) |
| `--box_model` / `--pose_model` | `yolo26m.pt` / `yolo26m-pose.pt` | `.pt` ou `.engine` TensorRT |
| `--force_court` | non | refaire la détection du court même si `calibration.json` existe |
| `--debug_video` | non | écrire `debug.mp4` (voir ci-dessous) ; écrit aussi `data_debug.json` |
| `--debug_max_side` | 1600 | plus grand côté (px) de la vue principale de la vidéo de debug |

## Vidéo de debug

`--debug_video` produit `outputs/<nom de la vidéo>/debug.mp4` (une image par frame traitée, lue à vitesse réelle) :

Le décodage, le rendu et l'encodage tournent en parallèle (~100 images/s en 1080p sur 16 coeurs). L'encodage est en H.264
via `ffmpeg` s'il est dans le PATH (rapide, fichiers ~10x plus petits), sinon en `mp4v` via OpenCV (plus lent, plus gros).

- **vue principale** : la vidéo d'origine (non redressée, avec sa distorsion) sur laquelle sont reprojetées, à travers
  le fisheye, les lignes du terrain calculées (contour blanc dont la ligne du fond, filet orange, lignes de service
  cyan). Si elles ne recouvrent pas les lignes peintes, la calibration est à revoir.
- **chaque détection** : box telle que détectée, id corrigé en petit au-dessus (avec `raw N` si l'id du tracker a été
  réassocié), trajectoire des 2 dernières secondes. La position en mètres est dans le panneau latéral.
- **pieds** : rond plein = centre bas de la box ; carré = milieu des deux chevilles projeté au sol (croix verte = milieu,
  chevilles peu fiables en rond creux) ; losange = estimé depuis le nez (point magenta). Dans les deux derniers cas, le rond
  creux est le centre bas de la box qui aurait été pris sinon.
- **détections ignorées** (box grise) avec la raison : pas de pose, ni cheville ni nez, hors terrain, pas d'id.
- **bandeau** : frame, temps, nombre de détections et de joueurs ; passe en rouge s'il n'y a pas exactement 4 joueurs
  ou si un id corrigé apparaît deux fois.
- **panneau latéral** : mini-carte du terrain vu de dessus, liste des joueurs (id corrigé, id brut, position,
  méthode), détections ignorées, et dernières réassociations d'ids.

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
