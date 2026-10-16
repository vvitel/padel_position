import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def plot_data(data, seq_keep):

    #créer la direction
    os.makedirs("plot", exist_ok=True)

    #changer valeurs des id temporel
    id_player_seq = np.unique(data[:, -1])
    data[:, -1] = np.searchsorted(id_player_seq, data[:, -1])

    frame_start = int(data[:, 0].min())
    frame_stop = int(data[:, 0].max())
    colors = {0: "darkblue", 1: "darkorange",
              2: "darkgreen", 3: "crimson",
              4: "deepskyblue", 5: "gold",
              6: "magenta", 7: "limegreen",
              8: "lightcoral", 9: "lightskyblue",
              10: "lightseagreen", 11: "goldenrod",
              12: "slateblue", 13: "chocolate",
              14: "springgreen", 15: "mediumvioletred"}

    for i in tqdm(range(frame_start, frame_stop)):
        if any(debut <= i <= fin for debut, fin in seq_keep):
            array_frame = data[data[:, 0] == i]
            for id_, color in colors.items():
                mask = array_frame[:, -1] == id_
                plt.scatter(array_frame[mask, 1], array_frame[mask, 2], color=color, label=str(id_))
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
            plt.savefig(f"plot/{i}")
            plt.close()
            
        else:
            plt.text(0, 10, "PAS DE JEU", fontsize=16)
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
            plt.savefig(f"plot/{i}")
            plt.close()
