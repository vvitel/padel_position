import numpy as np
from sklearn.cluster import DBSCAN, KMeans

def clustering(array, window_size, dbscan_eps, dbscan_min_samples):
    #frame de début et de fin
    frame_start = int(array[:, 0].min())
    frame_stop = int(array[:, 0].max())

    #découper en fenêtres de taille window_size
    lst_bloc = []
    for window_start in range(frame_start, frame_stop + 1, window_size):
        window_end = window_start + window_size
        bloc = array[(array[:, 0] >= window_start) &(array[:, 0] < window_end)]
        X = bloc[:, 1:3]   

        #si pas de détection
        if len(X) == 0:
            continue

        #clustering dbscan
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = dbscan.fit_predict(X)

        bloc = np.column_stack((bloc, labels))
        #supprimer les points non attribués
        bloc = bloc[bloc[:, -1] != -1]

        #compter nombre de points par cluster
        clusters, counts = np.unique(bloc[:, -1].astype(int), return_counts=True)
        counts_by_cluster = dict(zip(clusters, counts))

        #identifier clusters contenant trop de points
        cluster_merge = [
            cluster for cluster, count in counts_by_cluster.items()
            if window_size < count <= window_size*2
        ]

        #si trop de points pour un seul joueur
        if len(cluster_merge) != 0:
            for c in cluster_merge:
                rows = bloc[bloc[:, -1] == c]
                X = rows[:, 1:3]
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels_kmeans = kmeans.fit_predict(X)
                bloc[bloc[:, -1] == c, -1] = labels_kmeans + 777

        lst_bloc.append(bloc)

    #reformater les données en array
    array = np.vstack(lst_bloc)
    return array








         


