import numpy as np

def get_chains(correspondence):
    mapping = {int(a): int(b) for a, b in correspondence}
    chains, used = [], set()

    for a, _ in correspondence:
        current = int(a)
        if current in used:
            continue

        chain, local = [], set()
        while current in mapping and current not in local:
            chain.append(current)
            used.add(current)
            local.add(current)
            current = mapping[current]

        if current not in local:
            chain.append(current)
            used.add(current)

        chains.append(chain)

    return chains

def reassociation(data, life_of_ids):
    #identifier les zones de disparition
    lst_correspondence = []
    for life in life_of_ids:
        ids_disappears, frame_disappears = int(life[0]), int(life[2])
        if frame_disappears == -1: continue

        subset_disappears = data[(data[:, 0] == frame_disappears) & (data[:, 1] == ids_disappears)]
        x_disappears, y_disappears = subset_disappears[0, 2], subset_disappears[0, 3]

        #si disparition zone d'entrée du court
        if 9 <= y_disappears <= 11 and (0 <= x_disappears <= 1 or 9 <= x_disappears <= 10):
            continue

        #si disparition dans le court
        else:
            candidates = life_of_ids[life_of_ids[:, 1] > frame_disappears - 30]
            if len(candidates) > 0:
                #on test 3 apparitions temporellement les plus proches et au dessus
                closest_indices = np.argsort(np.abs(candidates[:, 1] - frame_disappears))[:3]
                closest_appears = candidates[closest_indices]

                for appears in closest_appears:
                    ids_appears, frame_appears = int(appears[0]), int(appears[1])
                    subset_appears = data[(data[:, 0] == frame_appears) & (data[:, 1] == ids_appears)]

                    if len(subset_appears) > 0:
                        x_appears, y_appears = subset_appears[0, 2], subset_appears[0, 3]
                        distance = np.sqrt((x_disappears - x_appears)**2 + (y_disappears - y_appears)**2)
                        if distance < 1.5:
                            lst_correspondence.append((ids_disappears, ids_appears))
                            break

    #lier les ids malgré les disparitions
    correspondence = np.array(lst_correspondence)
    correspondence = correspondence[correspondence[:, 0] != correspondence[:, 1]]
    chains  = get_chains(correspondence)
    mapping = {id_: chain[0] for chain in chains for id_ in chain}

    #remplacer les ids dans data
    data[:, 1] = np.array([mapping.get(int(id_), int(id_))for id_ in data[:, 1]])

    return data