import os
import json
from temporal_association.usefull_functions.retrieve_data import retrieve_data
from temporal_association.usefull_functions.identify_breaks import identify_breaks
from temporal_association.usefull_functions.reassociation import reassociation

def connect_players(out_dir=".", fps=30.0):
    path = os.path.join(out_dir, "data_homography.json")

    #lire les données
    data = retrieve_data(path)

    #identifier les ruptures dans le tracking
    life_of_ids = identify_breaks(data)

    #réassocier des ids malgré les disparitions
    mapping = reassociation(data, life_of_ids, fps=fps)

    #réécrire les données au format de data_homography.json avec les ids réassociés
    with open(path, "r") as fp:
        dico = json.load(fp)
    dico_tracks = {frame: [(mapping.get(track_id, track_id), x, y) for track_id, x, y in humans]
                   for frame, humans in dico.items()}

    with open(os.path.join(out_dir, "data_tracks.json"), "w") as fp:
        json.dump(dico_tracks, fp)

    return mapping
