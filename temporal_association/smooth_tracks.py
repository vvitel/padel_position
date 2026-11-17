import os
import json
import numpy as np

#lissage des trajectoires par id corrigé : filtre de Kalman à vitesse constante (avant) + lisseur RTS (arrière)
#- on a toute la vidéo, donc on lisse hors ligne : chaque position profite des mesures passées ET futures, sans retard
#- les trous courts (<= MAX_GAP_S) sont comblés par le modèle, les longs coupent la trajectoire en segments
#- les mesures aberrantes (saut de méthode bbox/chevilles/nez, mauvaise détection) sont rejetées par un test de
#  Mahalanobis sur l'innovation, sauf si elles s'enchaînent (le joueur a vraiment bougé)

SIGMA_ACC = 4.0           #écart-type de l'accélération (m/s²) : plus grand = suit mieux les changements de direction
SIGMA_MEAS = 0.25         #bruit de mesure de la position projetée (m)
SIGMA_V0 = 3.0            #incertitude sur la vitesse au début d'un segment (m/s)
GATE = 13.8               #seuil de rejet sur la distance de Mahalanobis² (chi² à 2 ddl, 99.9 %)
MAX_REJECT = 3            #au-delà de MAX_REJECT rejets consécutifs, la mesure est acceptée (re-accrochage)
MAX_GAP_S = 1.0           #trou plus long que ça : nouveau segment, pas d'interpolation

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
R = SIGMA_MEAS ** 2 * np.eye(2)


def _transition(dt):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = SIGMA_ACC ** 2
    a, b, c = q * dt ** 3 / 3, q * dt ** 2 / 2, q * dt
    Q = np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])
    return F, Q


def _smooth_segment(times, meas):
    """times : (n,) instants en secondes de la grille (toutes les frames traitées du segment)
    meas : liste de n listes de mesures (x, y) (vide = pas de mesure à cet instant)
    retour : (n, 4) états lissés [x, y, vx, vy] et (n,) booléen "au moins une mesure acceptée" """
    n = len(times)
    xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))   #prédictions
    xf, Pf = np.zeros((n, 4)), np.zeros((n, 4, 4))   #filtrés
    Fs = np.zeros((n, 4, 4))
    used = np.zeros(n, bool)

    x = np.r_[meas[0][0], 0.0, 0.0]
    P = np.diag([SIGMA_MEAS ** 2, SIGMA_MEAS ** 2, SIGMA_V0 ** 2, SIGMA_V0 ** 2])
    rejects = 0
    for k in range(n):
        if k > 0:
            F, Q = _transition(times[k] - times[k - 1])
            x, P = F @ x, F @ P @ F.T + Q
            Fs[k] = F
        xp[k], Pp[k] = x, P
        for z in meas[k]:
            innov = np.asarray(z) - H @ x
            S = H @ P @ H.T + R
            S_inv = np.linalg.inv(S)
            if k > 0 and innov @ S_inv @ innov > GATE and rejects < MAX_REJECT:
                rejects += 1
                continue
            rejects = 0
            K = P @ H.T @ S_inv
            x = x + K @ innov
            P = (np.eye(4) - K @ H) @ P
            used[k] = True
        xf[k], Pf[k] = x, P

    #lisseur RTS : on remonte le temps en corrigeant chaque état filtré avec l'état lissé suivant
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs, used


def smooth_tracks(out_dir=".", fps=30.0):
    """Lit data_tracks.json (ids corrigés) et écrit data_smooth.json :
    {frame: [[id, x, y, vx, vy, observed], ...]} en mètres et m/s ; observed = False pour une position
    comblée par le modèle (aucune mesure acceptée à cette frame)."""
    with open(os.path.join(out_dir, "data_tracks.json"), "r") as fp:
        dico = json.load(fp)

    #grille des frames traitées (y compris celles sans joueur) et mesures par id
    grid = np.array(sorted(int(f) for f in dico))
    tracks = {}
    for frame, humans in dico.items():
        for track_id, x, y in humans:
            if track_id is None: continue
            tracks.setdefault(int(track_id), {}).setdefault(int(frame), []).append((x, y))

    smooth = {int(f): [] for f in grid}
    max_gap = MAX_GAP_S * fps
    for track_id, by_frame in tracks.items():
        frames = np.array(sorted(by_frame))
        #segments : coupure là où le trou entre deux mesures dépasse MAX_GAP_S
        cuts = np.flatnonzero(np.diff(frames) > max_gap) + 1
        for seg in np.split(frames, cuts):
            steps = grid[(grid >= seg[0]) & (grid <= seg[-1])]
            xs, used = _smooth_segment(steps / fps, [by_frame.get(int(f), []) for f in steps])
            for f, (x, y, vx, vy), obs in zip(steps, xs, used):
                smooth[int(f)].append([track_id, float(x), float(y), float(vx), float(vy), bool(obs)])

    with open(os.path.join(out_dir, "data_smooth.json"), "w") as fp:
        json.dump(smooth, fp)
    return smooth
