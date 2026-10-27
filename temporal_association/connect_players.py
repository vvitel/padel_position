import numpy as np
from temporal_association.usefull_functions.retrieve_data import retrieve_data
from temporal_association.usefull_functions.clean_detections import clean_detections
from temporal_association.usefull_functions.identify_sequences import identify_sequences_to_keep
from temporal_association.usefull_functions.identify_breaks import identify_breaks
from temporal_association.usefull_functions.reassign_ids import reassign_ids

def connect_players():
    #lire les données
    data = retrieve_data()

    #nettoyer les détections
    data = clean_detections(data)

    #identifier les séquences de jeu
    seq_keep = identify_sequences_to_keep(data, 150)

    #identifier les ruptures dans le tracking
    lst_appearance, lst_disappearance = identify_breaks(data, seq_keep)

    #réassocier les id de tracking
    data = reassign_ids(data, seq_keep, lst_appearance, lst_disappearance)

    return data, seq_keep