import numpy as np
from temporal_association.usefull_functions.retrieve_data import retrieve_data
from temporal_association.usefull_functions.clustering import clustering
from temporal_association.usefull_functions.identify_no_game import identify_sequences_to_keep
from temporal_association.usefull_functions.temporal_association import temporal_association
from temporal_association.usefull_functions.kalman_filter import apply_kalman_filter
from temporal_association.usefull_functions.interp_data import interp_data

def connect_players():
    window_size = 15

    #lire les données
    data = retrieve_data()

    #clustering
    data = clustering(data, window_size, 0.75, 8)

    #identifier les séquences avec du jeu
    seq_keep = identify_sequences_to_keep(data, 150)

    #réaliser association temporelle par demi terrain
    related_data = temporal_association(data, seq_keep, window_size)

    #filtrer et interpoler les positions
    lst_array = []
    for s in np.unique(related_data[:, -1]):
        array_seq = related_data[related_data[:, -1] == s][:, :3]
        #gérer les données manquantes
        all_frames = np.arange(array_seq[:, 0].min(), array_seq[:, 0].max() + 1)
        missing_frames = all_frames[~np.isin(all_frames, array_seq[:, 0])]
        missing_rows = np.column_stack([missing_frames, np.full((len(missing_frames), 2), np.nan)])
        array_seq = np.vstack([array_seq, missing_rows])
        array_seq = array_seq[np.argsort(array_seq[:, 0])]
        #appliquer filtre de kalman
        array_seq_filter = apply_kalman_filter(array_seq)
        array_seq_filter = np.column_stack([array_seq_filter, np.full(len(array_seq_filter), s)])
        lst_array.append(array_seq_filter)

    #array filtrés et interpolés
    array_final = np.vstack(lst_array)
    return array_final, seq_keep