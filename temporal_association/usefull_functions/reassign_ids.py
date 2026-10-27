
import numpy as np

#reconstruit la trajectoire d'un joueur à travers plusieurs id
def path_finder(row_init, array_app_seq, array_disp_seq, ids_utilises, chemin=None):
    chemin = [] if chemin is None else chemin
    frame, id_courant = row_init[:2]
    y_mean, x_last, y_last = row_init[3:6]
    chemin.append(id_courant)
    ids_utilises.add(id_courant)

    #trier les apparitions suivantes par distance
    candidats = sorted(
        (row for row in array_app_seq if row[0] > frame),
        key=lambda row: np.hypot(row[4] - x_last, row[5] - y_last)
    )

    for row in candidats:
        new_id = row[1]
        meme_cote = ((y_mean > 10 and row[3] > 10)
                     or (y_mean < 10 and row[3] < 10))
        lignes = array_disp_seq[array_disp_seq[:, 1] == new_id]

        if new_id in ids_utilises or not meme_cote or not len(lignes):
            continue

        #tester récursivement la continuité de la trajectoire
        resultat = path_finder(
            lignes[0], array_app_seq, array_disp_seq,
            ids_utilises.copy(), chemin.copy()
        )

        if len(resultat) > len(chemin):
            ids_utilises.update(resultat)
            chemin[:] = resultat
            return chemin

    return chemin


#réattribue les IDs séparément pour chaque séquence conservée
def reassign_ids(data, seq_keep, lst_appearance, lst_disappearance):
    array_app = np.asarray(lst_appearance)
    array_disp = np.asarray(lst_disappearance)

    if array_app.size == 0 or array_disp.size == 0:
        return data

    for start, stop in np.asarray(seq_keep).reshape(-1, 2):
        start, stop = int(start), int(stop)
        mask_seq = (data[:, 0] >= start) & (data[:, 0] <= stop)
        data_seq = data[mask_seq]

        if not len(data_seq):
            continue

        #id présents au début de la séquence
        initial_ids = np.unique(data_seq[data_seq[:, 0] == start, 1])
        app_seq = array_app[
            (array_app[:, 0] >= start) & (array_app[:, 0] <= stop)
        ]
        disp_seq = array_disp[
            (array_disp[:, 0] >= start) & (array_disp[:, 0] <= stop)
        ]

        #ajoute les coordonnées moyennes et les points extrêmes
        def add_coords(rows, first):
            valid, coords = [], []

            for row in rows:
                d = data_seq[data_seq[:, 1] == row[1]]
                if not len(d):
                    continue

                valid.append(row)
                pos = d[0 if first else -1, 2:4]
                coords.append([d[:, 2].mean(), d[:, 3].mean(), *pos])

            return (np.column_stack((valid, coords)) if valid
                    else np.empty((0, rows.shape[1] + 4)))

        app_seq = add_coords(app_seq, True)
        disp_seq = add_coords(disp_seq, False)

        #sélectionne les disparitions des IDs initiaux
        disp_init = disp_seq[np.isin(disp_seq[:, 1], initial_ids)]
        disp_init = disp_init[np.argsort(-disp_init[:, 0])]
        disp_seq = disp_seq[~np.isin(disp_seq[:, 1], initial_ids)]

        #reconstruit les chemins des joueurs
        used, paths = set(), {}
        for row in disp_init:
            if row[1] not in used:
                paths[row[1]] = path_finder(row, app_seq, disp_seq, used)

        #conserve les id originaux de la séquence
        ids_originaux = data_seq[:, 1].copy()
        indices = np.where(mask_seq)[0]

        #regrouper tous les id présents dans les chemins
        ids_a_garder = set()
        for path in paths.values():
            ids_a_garder.update(path)

        #supprime les lignes dont l'ID n'appartient à aucun chemin
        mask_keep = np.isin(ids_originaux, list(ids_a_garder))
        data = np.delete(data, indices[~mask_keep], axis=0)

    return data
