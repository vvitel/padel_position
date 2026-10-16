import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

def interp_data(data, s):
    t = data[:, 0]
    t_unique, idx_unique = np.unique(t, return_index=True)
    x = data[idx_unique, 1]
    y = data[idx_unique, 2]
    t_new = np.arange(np.min(t), np.max(t), 1)

    interp_x = interp1d(t_unique, x, kind="linear")
    interp_y = interp1d(t_unique, y, kind="linear")
    x_new = interp_x(t_new)
    y_new = interp_y(t_new)
    #x_new = medfilt(x_new, kernel_size=5)
    #y_new = medfilt(y_new, kernel_size=5)
    cluster = np.repeat(s, len(t_new))
    array_final = np.column_stack((t_new, x_new, y_new, cluster))

    return array_final




