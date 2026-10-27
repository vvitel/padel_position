import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def clean_detections(data):
    #supprimer les points du côté et du fond
    lst_remove = []
    for i in np.unique(data[:, 1]):
        substr = data[data[:, 1] == i]
       
        #calcul du point moyen
        x_mean = np.mean(substr[:, 2])
        y_mean = np.mean(substr[:, 3])

        #côté gauche
        if (0 < x_mean < 0.5):
            lst_remove.append(i)
        #côté droit
        if (9.5 < x_mean < 10):
            lst_remove.append(i)
        #fond
        if (0 < y_mean < 0.5):
            lst_remove.append(i)

    data = data[~np.isin(data[:, 1], lst_remove)]

    #supprimer les détections trop proche
    lst_remove = []
    for i in np.unique(data[:, 0]):
        substr = data[data[:, 0] == i]
        for j in range(len(substr)):
            for k in range(j + 1, len(substr)):
                x1, y1 = substr[j, 2], substr[j, 3]
                x2, y2 = substr[k, 2], substr[k, 3]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                if distance < 0.3:
                    if substr[j, 1] > substr[k, 1]:
                        lst_remove.append(substr[j, 1])
                    else:
                        lst_remove.append(substr[k, 1])

    data = data[~np.isin(data[:, 1], lst_remove)]
    
    return data


def identify_incomplete_sequences(data, duration):
    start, stop = int(data[:, 0].min()), int(data[:, 0].max())
    frames = np.arange(start, stop + 1)
    bad = np.ones(len(frames), dtype=bool)

    for frame in frames:
        points = data[data[:, 0] == frame]
        y = points[:, 3]

        above = np.sum(y < 10)
        below = np.sum(y > 10)

        #si pas deux joueurs de chaque côtés
        bad[frame - start] = not (above == 2 and below == 2)

    #détecter les séquences continues de frames invalides
    changes = np.diff(np.r_[False, bad, False].astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    sequences = np.column_stack((frames[starts], frames[ends]))
    lengths = ends - starts + 1

    return sequences[lengths >= duration]

#identifier si joueur disparait dans un coin
def check_last_detected_position(data, seq_out, limit_zone_y):
    seq_out_2 = []
    for i in seq_out:
        previous_frame = int(i[0]) - 1
        if previous_frame in data[:, 0]:
            #récupérer les points de la frame précédente
            points = data[data[:, 0] == previous_frame, 2:4]
            y = points[:, 1]
            nb_points = np.sum(y > limit_zone_y)
            if nb_points != 2: seq_out_2.append(i)
    return seq_out_2

#définir les bornes séquences à traiter
def define_sequences_boundaries(frame_start, frame_stop, seq_out):
    lst_seq = []
    for i in range(len(seq_out)):
        if i == 0:
            seq = [frame_start, seq_out[i][0] - 1]
        else:
            seq = [seq_out[i - 1][1] + 1, seq_out[i][0] - 1]
        lst_seq.append(seq)
    lst_seq.append([seq_out[-1][1] + 1, frame_stop])
    return lst_seq

#regrouper les conditions
def identify_sequences_to_keep(data, duration):
    frame_start = int(data[:, 0].min())
    frame_stop = int(data[:, 0].max())
    seq_out = identify_incomplete_sequences(data, duration)
    if len(seq_out) == 0:
        seq_keep = [[frame_start, frame_stop]]
    else:
        seq_out_2 = check_last_detected_position(data, seq_out, 13)
        if len(seq_out_2) > 0:
            seq_keep = define_sequences_boundaries(frame_start, frame_stop, seq_out_2)
        else:
            seq_keep = [[frame_start, frame_stop]]
    return seq_keep

def link_sequences(data, seq_keep):
    for i in seq_keep:
        start, stop = int(i[0]), int(i[1])
        substr = data[(data[:, 0] >= start) & (data[:, 0] <= stop)]

        #tracking parfait
        if len(np.unique(substr[:, 1])) == 4:
            print("zizi")

        #tracking incomplet
        if len(np.unique(substr[:, 1])) != 4:
            frame_track_change = []
            for j in range(start+1, stop):
                prev_frame = np.sort(data[:, 1][data[:, 0] == j-1])
                curr_frame = np.sort(data[:, 1][data[:, 0] == j])

                if not np.array_equal(prev_frame, curr_frame):
                    frame_track_change.append(j)
            print(frame_track_change)

def identify_breaks(data, seq_keep):
    for i in seq_keep:
        start, stop = int(i[0]), int(i[1])
        substr = data[(data[:, 0] >= start) & (data[:, 0] <= stop)]

        #identifier les ruptures dans le tracking
        if len(np.unique(substr[:, 1])) != 4:
            frame_track_less = []
            for j in range(start+1, stop):
                prev_frame = np.sort(data[:, 1][data[:, 0] == j-1])
                curr_frame = np.sort(data[:, 1][data[:, 0] == j])

                if not np.array_equal(prev_frame, curr_frame):
                    if len(curr_frame) < len(prev_frame):
                        missing_ids = np.setdiff1d(prev_frame, curr_frame)
                        for k in missing_ids:
                            row = data[(data[:, 0] == j-1) & (data[:, 1] == k)].copy()
                            row[:, 0] += 1
                            data = np.vstack((data, row))
                        frame_track_less.append(j)

    data = data[np.argsort(data[:, 0])]
    return data


df = pd.read_csv("data_homography_video10.csv")
array = np.array(df)

"""
for i in np.unique(array[:, 1]):
    substr = array[array[:, 1] == i]
    print(len(substr))
    plt.scatter(substr[:, 2], substr[:, 3], color="k")
    plt.scatter(0, 0, c="pink")
    plt.scatter(10, 0, c="teal")
    plt.scatter(0, 20, c="teal")
    plt.scatter(10, 20, c="teal")
    plt.hlines(0, 0, 10, colors="k")
    plt.hlines(20, 0, 10, colors="k")
    plt.hlines(10, 0, 10, colors="k")
    plt.hlines(3.05, 0, 10, colors="k")
    plt.hlines(16.95, 0, 10, colors="k")
    plt.vlines(0, 0, 20, colors="k")
    plt.vlines(10, 0, 20, colors="k")
    plt.vlines(5, 3.05, 16.95, colors="k")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
"""

array = clean_detections(array)
seq_keep = identify_sequences_to_keep(array, 150)

"""
lst = []
for i in np.unique(array[:, 0]):
    substr = array[array[:, 0] == i]

    nb_player = np.unique(substr[:, 1])

    if len(nb_player) < 4:
        lst.append(i)
"""

link_sequences(array, seq_keep)