import os
import json
import numpy as np
from position_estimation.apply_homography import COURT_WIDTH, COURT_LENGTH
from temporal_association.usefull_functions.reassociation import in_door_zone

#lissage des trajectoires par id corrigé : filtre de Kalman (avant) + lisseur RTS (arrière)
#- on a toute la vidéo, donc on lisse hors ligne : chaque position profite des mesures passées ET futures, sans retard
#- modèle à vitesse "amortie" (Singer / Ornstein-Uhlenbeck) : la vitesse revient vers 0 avec la constante de temps TAU_V.
#  À l'échelle d'une frame c'est le modèle à vitesse constante ; sur un long trou, la vitesse ne part pas à l'infini
#  et le RTS relie les deux bords du trou par une courbe douce, sans dépassement (une vitesse constante donnerait
#  une cubique qui sort du terrain sur un trou de plusieurs secondes)
#- les trous sur le terrain sont comblés par le modèle ; la trajectoire est coupée (pas d'interpolation) quand le joueur
#  sort du court : vu hors du terrain pendant le trou (data_offcourt.json), ou trou de plus de DOOR_GAP_S qui commence
#  ou finit dans une zone de porte (il n'a pas été vu dehors, mais il est passé par la porte)
#- les mesures aberrantes (saut de méthode bbox/chevilles/nez, mauvaise détection) sont rejetées par un test de
#  Mahalanobis sur l'innovation, sauf si elles s'enchaînent (le joueur a vraiment bougé)

SIGMA_ACC = 4.0           #écart-type de l'accélération (m/s²) : plus grand = suit mieux les changements de direction
TAU_V = 0.5               #constante de temps (s) du retour de la vitesse vers 0 (plus petit = interpolation plus droite)
SIGMA_MEAS = 0.25         #bruit de mesure de la position projetée (m)
SIGMA_V0 = 3.0            #incertitude sur la vitesse au début d'un segment (m/s)
GATE = 13.8               #seuil de rejet sur la distance de Mahalanobis² (chi² à 2 ddl, 99.9 %)
MAX_REJECT = 3            #au-delà de MAX_REJECT rejets consécutifs, la mesure est acceptée (re-accrochage)
MAX_GAP_S = float("inf")  #trou plus long que ça : toujours coupé (inf = pas de limite, seule une sortie coupe)
OFF_COURT_MARGIN = 0.5    #un point hors terrain ne compte comme une sortie qu'au-delà de cette marge (m) : les pieds
                          #projetés juste derrière une ligne (vitre du fond, estimation par le nez) ne sont pas une sortie
DOOR_GAP_S = 1.0          #trou plus long que ça qui commence ou finit dans une zone de porte : sortie du court

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
R = SIGMA_MEAS ** 2 * np.eye(2)


def _transition(dt):
    #discrétisation exacte de dv = -v/TAU_V dt + SIGMA_ACC dW (même bruit que le modèle à vitesse constante quand
    #dt << TAU_V) ; e = 1 - exp(-dt/TAU_V)
    tau, q = TAU_V, SIGMA_ACC ** 2
    e = -np.expm1(-dt / tau)
    F = np.eye(4)
    F[0, 2] = F[1, 3] = tau * e
    F[2, 2] = F[3, 3] = 1 - e
    qpp = q * tau ** 2 * (dt - 2 * tau * e + tau * e * (2 - e) / 2)
    qpv = q * tau ** 2 * e ** 2 / 2
    qvv = q * tau * e * (2 - e) / 2
    Q = np.array([[qpp, 0, qpv, 0], [0, qpp, 0, qpv], [qpv, 0, qvv, 0], [0, qpv, 0, qvv]])
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


def _outside(x, y):
    #distance (m) au rectangle du terrain, 0 dedans
    return np.hypot(max(0.0, -x, x - COURT_WIDTH), max(0.0, -y, y - COURT_LENGTH))


def _load_exits(out_dir, mapping):
    #frames où chaque id corrigé est vu nettement hors du terrain : {id: tableau trié de frames}
    path = os.path.join(out_dir, "data_offcourt.json")
    if not os.path.exists(path): return {}
    with open(path, "r") as fp:
        dico = json.load(fp)
    exits = {}
    for frame, humans in dico.items():
        for track_id, x, y in humans:
            if track_id is None or _outside(x, y) <= OFF_COURT_MARGIN: continue
            exits.setdefault(mapping.get(int(track_id), int(track_id)), []).append(int(frame))
    return {k: np.array(sorted(v)) for k, v in exits.items()}


def _cuts(frames, by_frame, seen_out, fps):
    #indices i où la trajectoire est coupée entre frames[i-1] et frames[i] (le joueur est sorti du court)
    cuts = []
    for i in range(1, len(frames)):
        a, b = frames[i - 1], frames[i]
        gap = (b - a) / fps
        if gap > MAX_GAP_S: cuts.append(i); continue
        if seen_out.size and np.any((seen_out > a) & (seen_out < b)): cuts.append(i); continue
        if gap > DOOR_GAP_S and (in_door_zone(*np.mean(by_frame[a], axis=0)) or in_door_zone(*np.mean(by_frame[b], axis=0))):
            cuts.append(i)
    return cuts


def smooth_tracks(out_dir=".", fps=30.0, mapping=None):
    """Lit data_tracks.json (ids corrigés) et data_offcourt.json (ids bruts, traduits avec mapping) et écrit
    data_smooth.json : {frame: [[id, x, y, vx, vy, observed], ...]} en mètres et m/s ; observed = False pour une
    position comblée par le modèle (aucune mesure acceptée à cette frame). Un id est absent des frames où il est
    hors du court (et avant sa première / après sa dernière apparition)."""
    with open(os.path.join(out_dir, "data_tracks.json"), "r") as fp:
        dico = json.load(fp)
    exits = _load_exits(out_dir, mapping or {})

    #grille des frames traitées (y compris celles sans joueur) et mesures par id
    grid = np.array(sorted(int(f) for f in dico))
    tracks = {}
    for frame, humans in dico.items():
        for track_id, x, y in humans:
            if track_id is None: continue
            tracks.setdefault(int(track_id), {}).setdefault(int(frame), []).append((x, y))

    smooth = {int(f): [] for f in grid}
    for track_id, by_frame in tracks.items():
        frames = np.array(sorted(by_frame))
        #segments : coupure là où le joueur est sorti du court entre deux mesures
        cuts = _cuts(frames, by_frame, exits.get(track_id, np.empty(0, int)), fps)
        for seg in np.split(frames, cuts):
            steps = grid[(grid >= seg[0]) & (grid <= seg[-1])]
            xs, used = _smooth_segment(steps / fps, [by_frame.get(int(f), []) for f in steps])
            #toutes les mesures sont dans le terrain (in_court) : on y ramène aussi les positions comblées
            xs[:, 0] = np.clip(xs[:, 0], 0, COURT_WIDTH)
            xs[:, 1] = np.clip(xs[:, 1], 0, COURT_LENGTH)
            for f, (x, y, vx, vy), obs in zip(steps, xs, used):
                smooth[int(f)].append([track_id, float(x), float(y), float(vx), float(vy), bool(obs)])

    with open(os.path.join(out_dir, "data_smooth.json"), "w") as fp:
        json.dump(smooth, fp)
    return smooth
