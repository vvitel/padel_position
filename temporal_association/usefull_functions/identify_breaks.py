import numpy as np

def identify_breaks(data):
    start, stop = int(data[:, 0].min()), int(data[:, 0].max())
    substr = data[(data[:, 0] >= start) & (data[:, 0] <= stop)]

    #ids présents dès la première frame
    first_frame_ids = np.unique(substr[substr[:, 0] == start, 1])

    lst_appearance = [(start, int(id_)) for id_ in first_frame_ids]
    lst_disappearance = []

    #identifier les ruptures dans le tracking
    for j in range(start+1, stop):
        prev_frame = np.sort(substr[substr[:, 0] == j-1, 1])
        curr_frame = np.sort(substr[substr[:, 0] == j, 1])

        appeared_ids = np.setdiff1d(curr_frame, prev_frame)
        disappeared_ids = np.setdiff1d(prev_frame, curr_frame)

        #première apparition
        for id_ in appeared_ids:
            first_frame = substr[substr[:, 1] == id_, 0].min()
            if j == first_frame:
                lst_appearance.append((j, int(id_)))

        #dernière disparition
        for id_ in disappeared_ids:
            last_frame = substr[substr[:, 1] == id_, 0].max()
            if j-1 == last_frame:
                lst_disappearance.append((j-1, int(id_)))

    #construire le tableau final : ID | apparition | disparition
    all_ids = np.unique(substr[:, 1])
    result = np.full((len(all_ids), 3), -1, dtype=int)
    result[:, 0] = all_ids.astype(int)

    for frame, id_ in lst_appearance:
        result[result[:, 0] == id_, 1] = frame

    for frame, id_ in lst_disappearance:
        result[result[:, 0] == id_, 2] = frame

    return result