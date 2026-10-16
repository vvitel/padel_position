import numpy as np

def kalman(x, P, measurement, dt, R, q):
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    Q = q * np.array([[dt**4/4,0,dt**3/2,0],[0,dt**4/4,0,dt**3/2],[dt**3/2,0,dt**2,0],[0,dt**3/2,0,dt**2]])
    H = np.array([[1,0,0,0],[0,1,0,0]])

    x, P = F @ x, F @ P @ F.T + Q

    if measurement is not None:
        y = np.asarray(measurement).reshape(2) - H @ x
        K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
        x = x + K @ y
        A = np.eye(4) - K @ H
        P = A @ P @ A.T + K @ R @ K.T

    return x, P

def apply_kalman_filter(data):
    data = data[:, :3]
    X, P, R, q = np.zeros(4), np.eye(4)*1000, np.eye(2)*0.7**2, 4.0
    X[:2], previous_t = data[0, 1:3], data[0, 0]
    filtered = [(previous_t, X.copy())]

    for t, px, py in data[1:]:
        X, P = kalman(X, P, [px, py], t-previous_t, R, q)
        filtered.append((t, X.copy()))
        previous_t = t

    return np.array([[t, s[0], s[1]] for t, s in filtered])