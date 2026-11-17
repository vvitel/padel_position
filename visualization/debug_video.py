import os
import json
import math
import shutil
import subprocess
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from tqdm import tqdm
from position_estimation.apply_homography import image_to_world, in_court, COURT_WIDTH, COURT_LENGTH
from position_estimation.camera_functions.get_camera_parameters import get_camera_parameters
from position_estimation.players_detection import KEYPOINT_CONF

#vidéo de debug : la vidéo d'origine (distordue, non redressée) avec les lignes du terrain calculées reprojetées à
#travers le fisheye, les détections (box, id corrigé, pieds), un panneau latéral (mini-carte + infos) avec les
#positions brutes, et un second panneau avec les trajectoires lissées (data_smooth.json)

#géométrie du terrain en mètres (normes FIP, voir padel-court-detector/padelcourt/court.py)
NET_Y = COURT_LENGTH / 2
SERVICE_OFFSET = 6.95
CENTRE_X = COURT_WIDTH / 2
EXPECTED_PLAYERS = 4

#lignes du terrain : (point 1, point 2, couleur BGR, épaisseur)
WHITE, ORANGE, CYAN = (235, 235, 235), (0, 165, 255), (255, 220, 60)
COURT_LINES = [
    ((0, 0), (COURT_WIDTH, 0), WHITE, 2),
    ((COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH), WHITE, 2),
    ((COURT_WIDTH, COURT_LENGTH), (0, COURT_LENGTH), WHITE, 2),
    ((0, COURT_LENGTH), (0, 0), WHITE, 2),
    ((0, NET_Y), (COURT_WIDTH, NET_Y), ORANGE, 2),
    ((0, NET_Y - SERVICE_OFFSET), (COURT_WIDTH, NET_Y - SERVICE_OFFSET), CYAN, 1),
    ((0, NET_Y + SERVICE_OFFSET), (COURT_WIDTH, NET_Y + SERVICE_OFFSET), CYAN, 1),
    ((CENTRE_X, NET_Y - SERVICE_OFFSET), (CENTRE_X, NET_Y + SERVICE_OFFSET), CYAN, 1),
]

#un id corrigé = une couleur (palette de couleurs bien distinctes, en RGB)
PALETTE = [tuple(c[::-1]) for c in [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                                    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
                                    (0, 128, 128), (170, 110, 40)]]
GRAY = (150, 150, 150)

#raison pour laquelle une détection n'est pas utilisée (status de data_debug.json, puis cas détectés ici)
REASONS = {"no_crop": "no crop", "no_pose": "no pose", "no_nose": "no ankle/nose", "nose_failed": "nose ray failed",
           "off_court": "off court", "no_id": "no track id"}

PANEL_W = 440
TRAIL_SECONDS = 2.0
EVENTS_SHOWN = 6
PREFETCH = 6              #frames lues d'avance / écrites en attente (décodage et encodage tournent pendant le rendu)
LINE_SAMPLES = 400        #points échantillonnés par ligne du terrain (une droite du terrain devient courbe dans l'image)
BOTTOM, NOSE, ANKLE_L, ANKLE_R = 0, 1, 2, 3
ID_SCALE = 0.75           #taille de l'id des box, relative au texte normal (étiquette discrète)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _even(v):
    return int(math.ceil(v / 2) * 2)


def _project(hom, pts):
    #homographie sur des points (N, 2) -> points (N, 2) et dénominateurs
    p = np.c_[np.asarray(pts, dtype=float), np.ones(len(pts))] @ hom.T
    return p[:, :2] / p[:, 2:3], p[:, 2]


class DebugRenderer:
    def __init__(self, calibration, frame_size, max_side, video_fps, mapping, events, smooth=None):
        model, k, f, cx, cy, self.hom = get_camera_parameters(calibration)
        self.cam = {"model": model, "k": k, "f": f, "cx": cx, "cy": cy}
        self.hom_inv = np.linalg.inv(self.hom)
        self.mapping, self.events = mapping, events
        self.trail_frames = TRAIL_SECONDS * video_fps
        self.trails = {}
        self.smooth = smooth

        self._build_canvas(frame_size, max_side)
        self.fs = max(0.45, min(0.9, self.cw / 2200))
        self.thick = 1 if self.fs < 0.8 else 2
        self._build_court_layer()
        self._build_panel_background()
        if smooth is not None: self._build_smooth_trails(smooth)

    #---------- construction (une seule fois par vidéo) ----------

    def _build_canvas(self, frame_size, max_side):
        #le canevas est l'image d'origine (distordue), mise à l'échelle pour tenir dans max_side
        w_img, h_img = frame_size
        self.scale = min(1.0, max_side / float(max(w_img, h_img)))
        self.cw, self.ch = _even(w_img * self.scale), _even(h_img * self.scale)
        self.img = np.empty((self.ch, self.cw, 3), np.uint8)

        #le polynôme du fisheye n'est monotone que jusqu'à un certain angle : au-delà, la projection reboucle
        k1, k2 = self.cam["k"]
        theta = np.linspace(0, math.pi / 2, 20000, endpoint=False)
        bad = np.nonzero(np.diff(theta * (1 + k1 * theta ** 2 + k2 * theta ** 4)) <= 0)[0]
        self.theta_valid = theta[bad[0]] if bad.size else theta[-1]

    def _to_canvas(self, pts_image):
        #pixels de l'image d'origine -> pixels du canevas
        return np.asarray(pts_image, dtype=float) * self.scale

    def _distort(self, pts_undistorted):
        #pixels non distordus (ceux de l'homographie) -> pixels de l'image d'origine ; NaN au-delà de theta_valid
        f, cx, cy = self.cam["f"], self.cam["cx"], self.cam["cy"]
        norm = (np.asarray(pts_undistorted, dtype=np.float64).reshape(-1, 2) - (cx, cy)) / f
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array([*self.cam["k"], 0.0, 0.0], dtype=np.float64)
        out = cv2.fisheye.distortPoints(norm.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        out[np.arctan(np.hypot(norm[:, 0], norm[:, 1])) > self.theta_valid] = np.nan
        return out

    def _build_court_layer(self):
        #lignes du terrain calculées (homographie puis fisheye), dessinées une fois pour toutes ; une droite du terrain
        #est courbe dans l'image distordue, on l'échantillonne donc finement
        self.lines = np.zeros((self.ch, self.cw, 3), np.uint8)
        t = np.linspace(0, 1, LINE_SAMPLES)[:, None]
        for p1, p2, color, thickness in COURT_LINES:
            p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
            und, w = _project(self.hom, p1 + t * (p2 - p1))
            pts = self._to_canvas(self._distort(und))
            ok = (w > 1e-9) & np.isfinite(pts).all(axis=1)
            #une polyligne par morceau continu de points valides (derrière la caméra / hors du fisheye : coupé)
            for run in np.split(np.arange(len(pts)), np.flatnonzero(np.diff(ok.astype(int))) + 1):
                if ok[run[0]] and len(run) >= 2:
                    cv2.polylines(self.lines, [np.rint(pts[run]).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
        self.lines_idx = np.flatnonzero(self.lines.any(axis=2).ravel())
        self.lines_val = self.lines.reshape(-1, 3)[self.lines_idx]

    def _build_panel_background(self):
        #mini-carte du terrain vu de dessus (y=0 en haut = fond du court, comme sur la vidéo)
        self.panel_bg = np.full((self.ch, PANEL_W, 3), (28, 28, 28), np.uint8)
        extent_x, extent_y = COURT_WIDTH + 2, COURT_LENGTH + 2
        self.map_scale = min((PANEL_W - 20) / extent_x, (0.6 * self.ch - 20) / extent_y)
        self.map_origin = ((PANEL_W - extent_x * self.map_scale) / 2, 10.0)
        corner_a, corner_b = self._to_map((0, 0)), self._to_map((COURT_WIDTH, COURT_LENGTH))
        cv2.rectangle(self.panel_bg, corner_a, corner_b, (70, 55, 40), -1)
        for p1, p2, color, thickness in COURT_LINES:
            cv2.line(self.panel_bg, self._to_map(p1), self._to_map(p2), color, 1, cv2.LINE_AA)
        self.text_top = int(self.map_origin[1] + extent_y * self.map_scale + 22)

    def _to_map(self, world):
        return (int(round(self.map_origin[0] + (world[0] + 1) * self.map_scale)),
                int(round(self.map_origin[1] + (world[1] + 1) * self.map_scale)))

    #---------- rendu d'une frame ----------

    def render(self, frame, frame_index, fps, dets):
        if frame.shape[:2] == (self.ch, self.cw):
            np.copyto(self.img, frame)
            img = self.img
        else:
            img = cv2.resize(frame, (self.cw, self.ch), dst=self.img, interpolation=cv2.INTER_AREA)
        img.reshape(-1, 3)[self.lines_idx] = self.lines_val

        infos = [self._classify(d) for d in dets]
        pts = self._sample_points(infos)
        self._update_trails(frame_index, infos)
        self._draw_trails(img, frame_index)
        for info, p in zip(infos, pts): self._draw_detection(img, info, p)
        self._draw_legend(img)
        self._draw_hud(img, frame_index, fps, infos)
        panels = [img, self._draw_panel(frame_index, infos)]
        if self.smooth is not None: panels.append(self._draw_smooth_panel(frame_index))
        return np.hstack(panels)

    def _classify(self, d):
        raw = d["id"]
        cid = self.mapping.get(raw, raw) if raw is not None else None
        world = image_to_world(self.hom_inv, *d["foot"]) if d["foot"] else None
        if d["status"] != "ok": reason = d["status"]
        elif not in_court(*world): reason = "off_court"
        elif raw is None: reason = "no_id"
        else: reason = None
        return {"det": d, "raw": raw, "cid": cid, "world": world, "reason": reason}

    def _sample_points(self, infos):
        #pour chaque détection : centre bas de la box, nez, chevilles (image d'origine) en pixels entiers du canevas,
        #et pied estimé (en pixels non distordus, donc redistordu) rangé dans info["foot_canvas"]
        out = []
        for info in infos:
            d = info["det"]
            x1, _, x2, y2 = d["box"]
            bottom = ((x1 + x2) / 2, y2)
            #keypoints absents : on retombe sur le centre bas (point jamais dessiné)
            ankles = [a[:2] for a in d["ankles"] or []]
            pts = [bottom, d["nose"][:2] if d["nose"] else bottom] + (ankles + [bottom, bottom])[:2]
            info["foot_canvas"] = None
            if d["foot"]:
                foot = self._to_canvas(self._distort([d["foot"]]))[0]
                if np.isfinite(foot).all(): info["foot_canvas"] = np.rint(foot).astype(np.int32)
            out.append(np.rint(self._to_canvas(pts)).astype(np.int32).tolist())
        return out

    def _color(self, info):
        return PALETTE[info["cid"] % len(PALETTE)] if info["reason"] is None else GRAY

    def _update_trails(self, frame_index, infos):
        for info in infos:
            if info["reason"] is None and info["foot_canvas"] is not None:
                trail = self.trails.setdefault(info["cid"], deque(maxlen=int(self.trail_frames) + 1))
                trail.append((frame_index, info["foot_canvas"], self._to_map(info["world"])))

    def _recent(self, cid, frame_index):
        #(canevas, mini-carte) des points des TRAIL_SECONDS dernières secondes
        limit = frame_index - self.trail_frames
        return [(c, m) for fr, c, m in self.trails[cid] if fr >= limit]

    def _draw_trails(self, img, frame_index):
        for cid in self.trails:
            recent = self._recent(cid, frame_index)
            if len(recent) >= 2:
                cv2.polylines(img, [np.array([c for c, _ in recent], np.int32)], False, PALETTE[cid % len(PALETTE)], 2, cv2.LINE_AA)

    def _draw_detection(self, img, info, p):
        d, used = info["det"], info["reason"] is None
        color = self._color(info)
        r = max(4, int(7 * self.fs))

        #box telle que détectée dans l'image d'origine
        x1, y1, x2, y2 = np.rint(self._to_canvas(d["box"])).astype(int).tolist()
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if used else 1, cv2.LINE_AA)

        #keypoints de la pose (quand le modèle pose a tourné) : nez en magenta, chevilles visibles en vert
        if d["nose"] and d["nose"][2] > KEYPOINT_CONF:
            cv2.circle(img, tuple(p[NOSE]), max(3, r // 2), (255, 0, 255), -1, cv2.LINE_AA)
        for ankle, idx in zip(d["ankles"] or [], (ANKLE_L, ANKLE_R)):
            if ankle[2] > KEYPOINT_CONF:
                cv2.circle(img, tuple(p[idx]), max(3, r // 2), (0, 255, 0), -1, cv2.LINE_AA)
            elif d["method"] == "ankles":
                #cheville peu fiable mais quand même utilisée dans le milieu des deux chevilles : rond creux
                cv2.circle(img, tuple(p[idx]), max(3, r // 2), (0, 255, 0), 1, cv2.LINE_AA)

        #pieds tels qu'estimés par l'algorithme
        if info["foot_canvas"] is not None:
            foot = tuple(info["foot_canvas"].tolist())
            if d["method"] in ("nose", "ankles"):
                #estimation par rayon depuis un keypoint, reliée à lui ; rond creux = centre bas de la box (non utilisé)
                #nez : losange ; milieu des deux chevilles (croix verte) : carré
                if d["method"] == "nose":
                    source = tuple(p[NOSE])
                else:
                    source = ((p[ANKLE_L][0] + p[ANKLE_R][0]) // 2, (p[ANKLE_L][1] + p[ANKLE_R][1]) // 2)
                    cv2.line(img, tuple(p[ANKLE_L]), tuple(p[ANKLE_R]), (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.drawMarker(img, source, (0, 255, 0), cv2.MARKER_CROSS, r, 1, cv2.LINE_AA)
                cv2.line(img, source, foot, color, 1, cv2.LINE_AA)
                cv2.circle(img, tuple(p[BOTTOM]), r, color, 1, cv2.LINE_AA)
                if d["method"] == "nose":
                    shape = np.array([[foot[0], foot[1] - r - 2], [foot[0] + r + 2, foot[1]],
                                      [foot[0], foot[1] + r + 2], [foot[0] - r - 2, foot[1]]], np.int32)
                else:
                    shape = np.array([[foot[0] - r, foot[1] - r], [foot[0] + r, foot[1] - r],
                                      [foot[0] + r, foot[1] + r], [foot[0] - r, foot[1] + r]], np.int32)
                cv2.fillConvexPoly(img, shape, color, cv2.LINE_AA)
                cv2.polylines(img, [shape], True, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                #centre bas de la box : rond plein
                cv2.circle(img, foot, r, color, -1, cv2.LINE_AA)
                cv2.circle(img, foot, r, (0, 0, 0), 1, cv2.LINE_AA)

        #étiquette discrète : id seul, petit, sans fond, juste au-dessus du coin haut gauche de la box
        if used:
            text = str(info["cid"]) if info["raw"] == info["cid"] else f"{info['cid']} (raw {info['raw']})"
        else:
            text = f"{'-' if info['raw'] is None else info['raw']}: {REASONS[info['reason']]}"
        self._small_label(img, text, (x1 + 1, y1 - 3), color)

    def _small_label(self, img, text, org, color):
        #texte fin avec un liseré sombre (lisible sans rectangle de fond), recadré dans l'image
        scale = self.fs * ID_SCALE
        (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
        x = int(min(max(org[0], 0), max(0, img.shape[1] - tw - 2)))
        y = int(min(max(org[1], th + 2), img.shape[0] - base - 2))
        cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)

    def _label(self, img, text, org, color, scale=None):
        #texte lisible sur n'importe quel fond (rectangle noir derrière), recadré dans l'image
        scale = scale or self.fs
        (tw, th), base = cv2.getTextSize(text, FONT, scale, self.thick)
        x = int(min(max(org[0], 0), max(0, img.shape[1] - tw - 2)))
        y = int(min(max(org[1], th + 2), img.shape[0] - base - 2))
        cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + base), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 1, y - 1), FONT, scale, color, self.thick, cv2.LINE_AA)
        return th + base + 6

    def _draw_hud(self, img, frame_index, fps, infos):
        used = [i for i in infos if i["reason"] is None]
        cids = [i["cid"] for i in used]
        t = frame_index / fps
        text = f"frame {frame_index}  t={int(t // 60):02d}:{t % 60:04.1f}  detections {len(infos)}  players {len(set(cids))}/{EXPECTED_PLAYERS}"
        problem = len(set(cids)) != EXPECTED_PLAYERS or len(set(cids)) != len(cids)
        if len(set(cids)) != len(cids): text += "  DUPLICATE ID"
        self._label(img, text, (6, int(28 * self.fs) + 6), (80, 80, 255) if problem else (255, 255, 255), self.fs * 1.1)

    def _draw_legend(self, img):
        r = max(4, int(7 * self.fs))
        rows = [("foot = bbox bottom-centre", "circle"),
                ("foot = ground projection of ankles midpoint (cross)", "square"),
                ("foot = estimated from nose (hollow circle = bbox bottom)", "diamond"),
                ("magenta = nose, green = ankles (hollow = low confidence)", None),
                ("grey box = ignored detection (reason shown)", None),
                ("box label = corrected id (raw = tracker id before re-association)", None)]
        y = img.shape[0] - 6
        for text, icon in reversed(rows):
            h = self._label(img, text, (6 + 3 * r, y), (255, 255, 255))
            cy = y - h // 2 + 2
            if icon == "circle": cv2.circle(img, (6 + r, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
            if icon == "square":
                cv2.rectangle(img, (6, cy - r + 1), (6 + 2 * r, cy + r - 1), (255, 255, 255), -1)
            if icon == "diamond":
                cv2.fillConvexPoly(img, np.array([[6 + r, cy - r], [6 + 2 * r, cy], [6 + r, cy + r], [6, cy]], np.int32),
                                   (255, 255, 255), cv2.LINE_AA)
            y -= h

    def _draw_panel(self, frame_index, infos):
        panel = self.panel_bg.copy()

        #mini-carte : trajectoires récentes, joueurs retenus (plein) et détections hors terrain (creux gris)
        lines = [([m for _, m in self._recent(cid, frame_index)], PALETTE[cid % len(PALETTE)]) for cid in self.trails]
        points = [(info["world"], self._color(info), str(info["cid"]) if info["reason"] is None else None)
                  for info in infos if info["world"] is not None]
        self._draw_map(panel, lines, points)

        #liste texte
        row = self._text_rows(panel)
        used = sorted((i for i in infos if i["reason"] is None), key=lambda i: i["cid"])
        row("ON COURT  (#corrected id, raw tracker id)", (255, 255, 255))
        for i in used:
            x, yw = i["world"]
            row(f"#{i['cid']}  raw {i['raw']}  ({x:.1f},{yw:.1f})m  {i['det']['method']}", (230, 230, 230), self._color(i))
        if not used: row("none", GRAY)
        ignored = [i for i in infos if i["reason"] is not None]
        if ignored:
            row(None, None)
            row("IGNORED", (255, 255, 255))
            for i in ignored: row(f"raw {'-' if i['raw'] is None else i['raw']}: {REASONS[i['reason']]}", GRAY)
        past = [e for e in self.events if e[0] <= frame_index][-EVENTS_SHOWN:]
        row(None, None)
        row("RE-ASSOCIATIONS  (raw -> #id)", (255, 255, 255))
        for frame, raw, cid in past: row(f"frame {frame}: raw {raw} -> #{cid}", PALETTE[cid % len(PALETTE)])
        if not past: row("none yet", GRAY)
        return panel

    def _draw_map(self, panel, lines, points):
        #lines : [(points mini-carte, couleur)] ; points : [((x, y) en m, couleur, texte)]
        #texte None = rond creux (détection ignorée / position comblée), sinon rond plein avec l'id dedans
        radius = max(5, int(0.35 * self.map_scale))
        for line, color in lines:
            if len(line) >= 2: cv2.polylines(panel, [np.array(line, np.int32)], False, color, 1, cv2.LINE_AA)
        for (x, y), color, text in points:
            if not (-1 <= x <= COURT_WIDTH + 1 and -1 <= y <= COURT_LENGTH + 1): continue
            center = self._to_map((x, y))
            if text is None:
                cv2.circle(panel, center, radius, color, 1, cv2.LINE_AA)
                continue
            cv2.circle(panel, center, radius, color, -1, cv2.LINE_AA)
            cv2.circle(panel, center, radius, (0, 0, 0), 1, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
            cv2.putText(panel, text, (center[0] - tw // 2, center[1] + th // 2), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    def _text_rows(self, panel):
        #renvoie row(texte, couleur, pastille) qui écrit une ligne sous la mini-carte et descend ; row(None, None) = espace
        y = self.text_top
        def row(text, color, swatch=None):
            nonlocal y
            if text is None:
                y += 6
                return
            if y > self.ch - 6: return
            if swatch is not None: cv2.rectangle(panel, (10, y - 10), (20, y), swatch, -1)
            cv2.putText(panel, text, (28 if swatch is not None else 10, y), FONT, 0.5, color, 1, cv2.LINE_AA)
            y += 20
        return row

    #---------- panneau des trajectoires lissées ----------

    def _build_smooth_trails(self, smooth):
        #par id : frames et points de la mini-carte, coupés en morceaux là où le lissage a coupé la trajectoire
        #(l'id manque à une frame traitée de la grille = joueur sorti du court)
        rank = {frame: i for i, frame in enumerate(sorted(smooth))}
        per_id = {}
        for frame in sorted(smooth):
            for cid, x, y, *_ in smooth[frame]: per_id.setdefault(cid, []).append((frame, self._to_map((x, y))))
        self.smooth_trails = {}
        for cid, items in per_id.items():
            frames = np.array([f for f, _ in items])
            pts = np.array([m for _, m in items], np.int32)
            cuts = np.flatnonzero(np.diff([rank[f] for f in frames]) > 1) + 1
            self.smooth_trails[cid] = list(zip(np.split(frames, cuts), np.split(pts, cuts)))

    def _draw_smooth_panel(self, frame_index):
        panel = self.panel_bg.copy()
        limit = frame_index - self.trail_frames
        lines = []
        for cid, pieces in self.smooth_trails.items():
            for frames, pts in pieces:
                keep = (frames >= limit) & (frames <= frame_index)
                if keep.any(): lines.append((pts[keep], PALETTE[cid % len(PALETTE)]))
        players = sorted(self.smooth.get(frame_index, []))
        points = [((x, y), PALETTE[cid % len(PALETTE)], str(cid) if observed else None)
                  for cid, x, y, _, _, observed in players]
        self._draw_map(panel, lines, points)

        row = self._text_rows(panel)
        row("SMOOTHED  (Kalman + RTS, per corrected id)", (255, 255, 255))
        for cid, x, y, vx, vy, observed in players:
            speed = 3.6 * math.hypot(vx, vy)
            row(f"#{cid}  ({x:.1f},{y:.1f})m  {speed:4.1f} km/h{'' if observed else '  filled'}", (230, 230, 230),
                PALETTE[cid % len(PALETTE)])
        if not players: row("none", GRAY)
        row(None, None)
        row("hollow = no accepted measurement", GRAY)
        row("  (gap filled or outlier rejected)", GRAY)
        return panel


def load_debug(out_dir):
    path = os.path.join(out_dir, "data_debug.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable : detect_players doit être lancé avec debug=True")
    with open(path, "r") as fp:
        return {int(frame): dets for frame, dets in json.load(fp).items()}


def load_smooth(out_dir):
    #positions lissées (data_smooth.json, voir temporal_association/smooth_tracks.py) ; None si absentes
    path = os.path.join(out_dir, "data_smooth.json")
    if not os.path.exists(path): return None
    with open(path, "r") as fp:
        return {int(frame): players for frame, players in json.load(fp).items()}


def reassociation_events(dico, mapping):
    #première frame où apparaît chaque id brut qui a été rattaché à un id plus ancien
    first = {}
    for frame in sorted(dico):
        for d in dico[frame]:
            if d["id"] in mapping: first.setdefault(d["id"], frame)
    return sorted((frame, raw, mapping[raw]) for raw, frame in first.items())


class VideoSink:
    """Écrit des frames BGR. H.264 via ffmpeg (multi-thread, petits fichiers, lisible partout) s'il est dans le PATH,
    sinon mp4v d'OpenCV (monothread, bien plus lent et plus gros)."""

    def __init__(self, path, fps, size):
        self.path, self.proc, self.writer = path, None, None
        exe = shutil.which("ffmpeg")
        if exe:
            self.log = tempfile.TemporaryFile()
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                   "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        else:
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self.writer.isOpened(): raise RuntimeError(f"impossible d'écrire {path}")

    def write(self, img):
        if self.writer is not None:
            self.writer.write(img)
            return
        try:
            self.proc.stdin.write(img.data)
        except (BrokenPipeError, OSError):
            raise RuntimeError(f"ffmpeg a échoué : {self._error()}")

    def _error(self):
        self.log.seek(0)
        return self.log.read().decode(errors="replace").strip()

    def close(self):
        if self.writer is not None: self.writer.release()
        if self.proc is None: return
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        if self.proc.wait() != 0: raise RuntimeError(f"ffmpeg a échoué : {self._error()}")
        self.log.close()


def render_debug_video(video_path, out_dir, calibration, mapping, fps, step=1, max_side=1600):
    dico = load_debug(out_dir)
    frames = sorted(dico)
    if not frames: raise RuntimeError("data_debug.json est vide : aucune frame à afficher")
    wanted, last = set(frames), frames[-1]

    cap = cv2.VideoCapture(video_path)
    out_path = os.path.join(out_dir, "debug.mp4")
    sink = None
    try:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        smooth = load_smooth(out_dir)
        renderer = DebugRenderer(calibration, size, max_side, fps, mapping, reassociation_events(dico, mapping), smooth)
        #la vidéo de debug ne contient que les frames traitées : on la ralentit pour qu'elle reste en temps réel
        panels = 1 if smooth is None else 2
        sink = VideoSink(out_path, fps / max(step, 1), (renderer.cw + panels * PANEL_W, renderer.ch))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        state = {"index": frames[0]}

        def read_next():
            #prochaine frame à afficher (les autres sont sautées) ; un seul thread lecteur, donc l'ordre est garanti
            while state["index"] <= last:
                i = state["index"]
                state["index"] += 1
                if i in wanted:
                    ret, frame = cap.read()
                    return (i, frame) if ret else None
                if not cap.grab(): return None
            return None

        #décodage, rendu (ce thread) et encodage sont trois étapes en parallèle : OpenCV libère le GIL
        with ThreadPoolExecutor(1) as reader, ThreadPoolExecutor(1) as writer, tqdm(total=len(frames), desc="debug video") as pbar:
            reads = deque(reader.submit(read_next) for _ in range(PREFETCH))
            writes = deque()
            while reads:
                item = reads.popleft().result()
                if item is None: break
                reads.append(reader.submit(read_next))
                index, frame = item
                #result() relève les erreurs du thread d'écriture ; la file bornée limite la mémoire
                if len(writes) >= PREFETCH: writes.popleft().result()
                writes.append(writer.submit(sink.write, renderer.render(frame, index, fps, dico[index])))
                pbar.update(1)
            for fut in writes: fut.result()
    finally:
        cap.release()
        if sink is not None: sink.close()
    return out_path
