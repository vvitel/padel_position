import numpy as np
from temporal_association.usefull_functions.retrieve_data import retrieve_data
from temporal_association.usefull_functions.identify_breaks import identify_breaks
from temporal_association.usefull_functions.reassociation import reassociation

def connect_players():
    #lire les données
    data = retrieve_data()

    #identifier les ruptures dans le tracking
    life_of_ids = identify_breaks(data)

    #réassocier des ids malgré les disparitions
    new_data = reassociation(data, life_of_ids)
    np.save("new_data.npy", new_data)