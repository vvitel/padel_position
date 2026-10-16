import numpy as np

def identify_incomplete_sequences(data, duration):
    start, stop = int(data[:, 0].min()), int(data[:, 0].max())
    frames = np.arange(start, stop + 1)
    bad = np.ones(len(frames), dtype=bool)

    for frame in frames:
        points = data[data[:, 0] == frame]
        y = points[:, 2]

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
            points = data[data[:, 0] == previous_frame, 1:3]
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