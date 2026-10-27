import numpy as np

def clean_detections(data):
    #supprimer les points du côté et du fond
    lst_remove = []
    for i in np.unique(data[:, 1]):
        substr = data[data[:, 1] == i]
        x_mean = np.mean(substr[:, 2])
        y_mean = np.mean(substr[:, 3])
        if (0 < x_mean < 1):
            lst_remove.append(i)
        if (9 < x_mean < 10):
            lst_remove.append(i)
        if (0 < y_mean < 1):
            lst_remove.append(i)

    data = data[~np.isin(data[:, 1], lst_remove)]

    #supprimer les détections trop proches
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