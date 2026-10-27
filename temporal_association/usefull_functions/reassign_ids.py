
import numpy as np


def path_finder(row_init, array_app_seq, array_disp_seq, ids_utilises, chemin=None):
    if chemin is None:
        chemin = []

    frame, id_courant = row_init[:2]
    y_mean, x_last, y_last = row_init[3:6]
    chemin.append(id_courant)
    ids_utilises.add(id_courant)

    for row in array_app_seq[array_app_seq[:, 0] > frame]:
        new_id = row[1]
        if new_id in ids_utilises:
            continue

        meme_cote = (
            (y_mean > 10 and row[3] > 10)
            or (y_mean < 10 and row[3] < 10)
        )
        distance = np.hypot(row[4] - x_last, row[5] - y_last)

        if meme_cote and distance < 4:
            lignes = array_disp_seq[array_disp_seq[:, 1] == new_id]
            if len(lignes):
                return path_finder(
                    lignes[0], array_app_seq, array_disp_seq,
                    ids_utilises, chemin
                )

    return chemin


def reassign_ids(data, seq_keep, lst_appearance, lst_disappearance):
    array_app = np.asarray(lst_appearance)
    array_disp = np.asarray(lst_disappearance)

    if array_app.size == 0 or array_disp.size == 0:
        return data

    for i in seq_keep:
        start, stop = map(int, i[:2])

        # Données de la séquence uniquement
        mask_seq = (data[:, 0] >= start) & (data[:, 0] <= stop)
        data_seq = data[mask_seq]

        if len(data_seq) == 0:
            continue

        initial_ids = np.unique(data_seq[data_seq[:, 0] == start, 1])

        app_seq = array_app[
            (array_app[:, 0] >= start) & (array_app[:, 0] <= stop)
        ]
        disp_seq = array_disp[
            (array_disp[:, 0] >= start) & (array_disp[:, 0] <= stop)
        ]

        def add_coords(rows, first):
            valid_rows, coords = [], []

            for row in rows:
                data_id = data_seq[data_seq[:, 1] == row[1]]

                if len(data_id) == 0:
                    continue

                valid_rows.append(row)
                coords.append([
                    data_id[:, 2].mean(),
                    data_id[:, 3].mean(),
                    data_id[0 if first else -1, 2],
                    data_id[0 if first else -1, 3]
                ])

            if valid_rows:
                return np.column_stack((valid_rows, coords))

            return np.empty((0, rows.shape[1] + 4))

        array_app_seq = add_coords(app_seq, first=True)
        array_disp_seq = add_coords(disp_seq, first=False)

        disp_init_id = array_disp_seq[
            np.isin(array_disp_seq[:, 1], initial_ids)
        ]
        disp_init_id = disp_init_id[np.argsort(-disp_init_id[:, 0])]

        array_disp_seq = array_disp_seq[
            ~np.isin(array_disp_seq[:, 1], initial_ids)
        ]

        ids_utilises, all_path = set(), {}

        for row in disp_init_id:
            id_init = row[1]

            if id_init not in ids_utilises:
                all_path[id_init] = path_finder(
                    row, array_app_seq, array_disp_seq, ids_utilises
                )

        # Réattribuer les IDs dans cette séquence seulement
        for id_init, chemin in all_path.items():
            mask = (
                (data[:, 0] >= start)
                & (data[:, 0] <= stop)
                & np.isin(data[:, 1], chemin)
            )
            data[mask, 1] = id_init

    return data
