import numpy as np

def distort(xy, k):
    x = xy[:, 0]
    y = xy[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)

    theta_distorted = theta * (
        1 + k[0] * theta**2 + k[1] * theta**4
    )

    scale = np.ones_like(r)
    mask = r > 1e-12
    scale[mask] = theta_distorted[mask] / r[mask]

    return np.column_stack((
        x * scale,
        y * scale,
    ))