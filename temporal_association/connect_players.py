import matplotlib.pyplot as plt
import numpy as np
from temporal_association.usefull_functions.retrieve_data import retrieve_data
from temporal_association.usefull_functions.identify_breaks import identify_breaks

def connect_players():
    #lire les données
    data = retrieve_data()
    start, stop = min(data[:, 0]), max(data[:, 0])

    #identifier les ruptures dans le tracking
    life_of_ids = identify_breaks(data)
    print(life_of_ids)

    for i in np.unique(data[:, 1]):
        print(i)
        substr = data[data[:, 1] == i]
        plt.scatter(substr[:, 2], substr[:, 3])
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
        plt.xlim(0, 10) 
        plt.ylim(-1, 21)
        plt.show()
        plt.close()


