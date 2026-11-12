import numpy as np

def identify_breaks(data):
    #pour chaque id : première et dernière frame où il est vu
    #retour : tableau ID | apparition | disparition
    #apparition = -1 si l'id est déjà là à la première frame des données (pas une vraie apparition)
    #disparition = -1 si l'id est encore là à la dernière frame des données (pas une vraie disparition)
    if len(data) == 0: return np.empty((0, 3), dtype=int)
    start, stop = int(data[:, 0].min()), int(data[:, 0].max())

    ids, inverse = np.unique(data[:, 1].astype(int), return_inverse=True)
    frames = data[:, 0].astype(int)
    first = np.full(len(ids), np.iinfo(int).max)
    last = np.full(len(ids), np.iinfo(int).min)
    np.minimum.at(first, inverse, frames)
    np.maximum.at(last, inverse, frames)

    result = np.stack([ids, first, last], axis=1)
    result[first == start, 1] = -1
    result[last == stop, 2] = -1

    return result
