import numpy as np

def identify_breaks(data, seq_keep):
    for i in seq_keep:
        start, stop = int(i[0]), int(i[1])
        substr = data[(data[:, 0] >= start) & (data[:, 0] <= stop)]
        lst_appearance, lst_disappearance = [], []

        #identifier les ruptures dans le tracking
        if len(np.unique(substr[:, 1])) != 4:
            for j in range(start+1, stop):
                prev_frame = np.sort(substr[:, 1][substr[:, 0] == j-1])
                curr_frame = np.sort(substr[:, 1][substr[:, 0] == j])
                appeared_ids = np.setdiff1d(curr_frame, prev_frame)
                diappeared_ids = np.setdiff1d(prev_frame, curr_frame)

                #première apparition
                for id_ in appeared_ids:
                    first_frame = substr[substr[:, 1] == id_, 0].min()
                    if j == first_frame:
                        lst_appearance.append((j, int(id_)))

                #dernière disparition
                for id_ in diappeared_ids:
                    last_frame = substr[substr[:, 1] == id_, 0].max()
                    if j-1 == last_frame:
                        lst_disappearance.append((j-1, int(id_)))

    return lst_appearance, lst_disappearance