import numpy as np

def identify_breaks(data):
    frames = data[:, 0].astype(int)
    ids = data[:, 1].astype(int)

    # Compresser les IDs pour obtenir des indices contigus
    all_ids, inverse = np.unique(ids, return_inverse=True)

    # Première et dernière frame de chaque ID
    first = np.full(len(all_ids), np.iinfo(np.int64).max, dtype=np.int64)
    last = np.full(len(all_ids), -1, dtype=np.int64)

    np.minimum.at(first, inverse, frames)
    np.maximum.at(last, inverse, frames)

    # Résultat : ID | apparition | disparition
    result = np.full((len(all_ids), 3), -1, dtype=int)
    result[:, 0] = all_ids
    result[:, 1] = first

    # Une disparition n'est retenue que si l'ID
    # ne réapparaît pas après une interruption.
    result[:, 2] = last

    return result
