import json
import numpy as np

def retrieve_data():
    with open("data_homography.json", "r") as f:
        dico = json.load(f)

    #formater les données en array
    array = []
    for frame, humans in dico.items():
        for _, position in humans.items():
            x, y = position
            array.append([int(frame), x, y])
    array = np.array(array, dtype=np.float64)

    return array