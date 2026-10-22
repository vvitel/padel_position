import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def temporal_association(data, seq_keep, window_size):
    #séparer terrain du haut et du bas
    data_up = data[data[:, 2] < 9.7]
    data_down = data[data[:, 2] > 10.3]
    cpt_semi_court = 0

    lst_final_temporal = []
    for a in [data_up, data_down]:
        for seq in seq_keep:
            frame_start, frame_stop = map(int, seq)
            lst_center, lst_data = [], []

            #on redécoupe en fênetres de plusieurs frame
            for window_start in range(frame_start, frame_stop + 1, window_size):
                window_end = window_start + window_size
                bloc = a[(a[:, 0] >= window_start) & (a[:, 0] < window_end)].copy()

                #si vide ou pas deux clusters on passe
                if len(bloc) == 0 or len(np.unique(bloc[:, -1])) != 2:
                    continue

                #récupérer point moyen par cluster
                for c in np.unique(bloc[:, -1]):
                    points_cluster = bloc[bloc[:, -1] == c]
                    centre_x = np.mean(points_cluster[:, 1])
                    centre_y = np.mean(np.mean(points_cluster[:, 2]))

                    #stocker le centre et l'identifiant du cluster original
                    lst_center.append([window_start, centre_x, centre_y, c, -1])
                    #stocker tous les points du cluster
                    lst_data.append((window_start, c, points_cluster.copy()))

            #reformater en array
            array_center = np.array(lst_center, dtype=float)
            if len(array_center) == 0: continue
            frames = np.unique(array_center[:, 0])

            #début de l'association temporelle
            #initialisation des id temporels
            idx_init = np.where(array_center[:, 0] == frames[0])[0]
            array_center[idx_init, 4] = array_center[idx_init, 3]

            #parcours des frames
            for f in range(1, len(frames)):
                frame_t = frames[f - 1]
                frame_t1 = frames[f]

                idx_t = np.where((array_center[:, 0] == frame_t))[0]
                idx_t1 = np.where((array_center[:, 0] == frame_t1))[0]

                if len(idx_t) == 0 or len(idx_t1) == 0: continue

                #calcul des distances entre les centres
                centres_t = array_center[idx_t, 1:3]
                centres_t1 = array_center[idx_t1, 1:3]
                distances = cdist(centres_t, centres_t1)
                indices_t, indices_t1 = linear_sum_assignment(distances)

                for i, j in zip(indices_t, indices_t1):
                    array_center[idx_t1[j], 4] = array_center[idx_t[i], 4]

            #reporter les ids temporels sur toutes les données
            lst_results = []
            for window_start, cluster_original, points_cluster in lst_data:
                idx = np.where((array_center[:, 0] == window_start) & (array_center[:, 3] == cluster_original))[0]
                if len(idx) == 0: continue
                id_temporel = array_center[idx[0], 4]
                
                #ajouter l'id temporel comme dernière colonne
                points_avec_id = np.column_stack([points_cluster, np.full(len(points_cluster), id_temporel)])
                lst_results.append(points_avec_id)
            
            array_results = np.vstack(lst_results)
            array_results[:, -1] = array_results[:, -1] + cpt_semi_court
            cpt_semi_court += 10
            lst_final_temporal.append(array_results)

    #reformater
    array_final = np.vstack(lst_final_temporal)
    return array_final


  


      










