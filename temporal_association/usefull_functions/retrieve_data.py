import json
import numpy as np

def retrieve_data(path="data_homography.json"):
    with open(path, "r") as f:
        dico = json.load(f)

    array = []

    for frame, humans in dico.items():
        for person in humans:
            track_id, x, y = person
            array.append([int(frame), int(track_id), x, y])

    array = np.array(array, dtype=np.float64)

    return array