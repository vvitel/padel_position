import numpy as np
from scipy.optimize import linear_sum_assignment

#zones des portes (côtés du court au niveau du filet) : un joueur qui y disparaît sort du court
DOOR_Y = (9.0, 11.0)
DOOR_X = ((0.0, 1.0), (9.0, 10.0))
NET_Y = 10.0


def in_door_zone(x, y):
    return DOOR_Y[0] <= y <= DOOR_Y[1] and any(a <= x <= b for a, b in DOOR_X)


def position_at(data, track_id, frame):
    row = data[(data[:, 0] == frame) & (data[:, 1] == track_id)]
    return row[0, 2], row[0, 3]


def reassociation(data, life_of_ids, fps=30.0, max_gap_s=5.0, base_dist=1.0, max_speed=6.0, net_margin=0.5):
    """Relie chaque piste qui disparaît à au plus une piste qui apparaît après elle.

    Une liaison fin(A) -> début(B) est possible si :
      - B commence strictement après la fin de A (pas de chevauchement : deux joueurs visibles en même temps
        ne peuvent pas être fusionnés)
      - l'écart dt est inférieur à max_gap_s
      - la distance est inférieure à base_dist + max_speed * dt (le rayon grandit avec le temps d'absence)
      - A et B sont du même côté du filet (sauf si l'un des deux est à moins de net_margin du filet)
      - A ne disparaît pas par une porte (le joueur sort du court)
    Les liaisons sont choisies globalement et une à une (algorithme hongrois), sur la distance normalisée
    par le rayon autorisé. Les liaisons successives forment des chaînes A -> B -> C.

    Retour : dict {ancien id: nouvel id} (le nouvel id est le premier id de la chaîne)
    """
    #fins et débuts de pistes avec leur position
    ends = []
    for track_id, _, frame_end in life_of_ids:
        if frame_end == -1: continue
        x, y = position_at(data, track_id, frame_end)
        if in_door_zone(x, y): continue
        ends.append((int(track_id), int(frame_end), x, y))

    starts = []
    for track_id, frame_start, _ in life_of_ids:
        if frame_start == -1: continue
        x, y = position_at(data, track_id, frame_start)
        starts.append((int(track_id), int(frame_start), x, y))

    if not ends or not starts: return {}

    #matrice des coûts
    INF = 1e6
    max_gap = max_gap_s * fps
    cost = np.full((len(ends), len(starts)), INF)
    for i, (id_a, f_a, x_a, y_a) in enumerate(ends):
        for j, (id_b, f_b, x_b, y_b) in enumerate(starts):
            dt = f_b - f_a
            if id_a == id_b or dt <= 0 or dt > max_gap: continue
            if abs(y_a - NET_Y) > net_margin and abs(y_b - NET_Y) > net_margin and (y_a < NET_Y) != (y_b < NET_Y):
                continue
            radius = base_dist + max_speed * dt / fps
            dist = np.hypot(x_a - x_b, y_a - y_b)
            if dist <= radius: cost[i, j] = dist / radius

    #affectation un pour un
    rows, cols = linear_sum_assignment(cost)
    successor = {ends[i][0]: starts[j][0] for i, j in zip(rows, cols) if cost[i, j] < INF}

    #chaînes : chaque id a au plus un successeur et un prédécesseur, et chaque liaison va vers le futur
    #donc pas de cycle ; la racine d'une chaîne est un id qui n'est le successeur de personne
    has_predecessor = set(successor.values())
    mapping = {}
    for root in successor:
        if root in has_predecessor: continue
        current = root
        while current in successor:
            current = successor[current]
            mapping[current] = root

    return mapping

