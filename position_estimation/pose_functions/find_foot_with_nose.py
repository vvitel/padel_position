import numpy as np

def camera_pose_from_homography(H, f, cx, cy):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    #retirer les paramètres intrinsèques de l'homographie
    B = np.linalg.inv(K) @ H

    b1, b2, b3 = B[:, 0], B[:, 1], B[:, 2]

    #retrouver l'échelle et les vecteurs de rotation
    scale = 2 / (np.linalg.norm(b1) + np.linalg.norm(b2))
    r1, r2, t = scale * b1, scale * b2, scale * b3

    #le terrain doit être devant la caméra (H n'est défini qu'à un facteur près, signe compris)
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t

    #construire une matrice de rotation
    R_approx = np.column_stack((r1, r2, np.cross(r1, r2)))

    #corriger les erreurs numériques pour obtenir une rotation valide
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    #calculer la position de la caméra dans le repère terrain
    C = -R.T @ t

    return R, C

def find_foot_with_nose(x_nose, y_nose, H, f, cx, cy, h):
    R, C = camera_pose_from_homography(H, f, cx, cy)

    #construire le rayon caméra passant par le nez
    ray = np.array([(x_nose-cx)/f, (y_nose-cy)/f, 1.0])
    D = R.T @ ray

    #trouver le point du rayon à la hauteur h
    if abs(D[2]) < 1e-8:
        return None

    #le sens de l'axe Z (r1 x r2) dépend de l'orientation des axes du terrain :
    #la hauteur h se compte du côté du sol où se trouve la caméra
    z_nose = np.sign(C[2]) * h
    t = (z_nose - C[2]) / D[2]
    if t <= 0:
        return None

    nose_3d = C + t * D

    #projection verticale sur le sol puis retour dans l'image (pixels non distordus),
    #pour rester cohérent avec les autres positions transformées par apply_homography
    foot_img = H @ np.array([nose_3d[0], nose_3d[1], 1.0])
    return float(foot_img[0] / foot_img[2]), float(foot_img[1] / foot_img[2])

