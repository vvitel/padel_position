
import matplotlib.pyplot as plt
import numpy as np
import os


def create_plot(data, seq_keep):
    lst_color = [
        "red", "blue", "orange", "green", "purple", "brown",
        "pink", "gray", "olive", "cyan", "gold"
    ]

    # Convertir les frames et les intervalles en valeurs numériques
    frames = data[:, 0].astype(float)
    seq_keep = np.asarray(seq_keep, dtype=float).reshape(-1, 2)

    # Associer une couleur à chaque ID
    ids = np.unique(data[:, 1])
    color_by_id = {
        id_: lst_color[i % len(lst_color)]
        for i, id_ in enumerate(ids)
    }

    os.makedirs("plot", exist_ok=True)

    for f in np.unique(frames):
        fig, ax = plt.subplots()

        # Vérifier si la frame appartient à un intervalle conservé
        en_jeu = any(debut <= f <= fin for debut, fin in seq_keep)

        if en_jeu:
            subset = data[frames == f]

            for row in subset:
                ax.scatter(
                    float(row[2]),
                    float(row[3]),
                    color=color_by_id[row[1]]
                )
        else:
            ax.text(
                5, 10, "PAS DE JEU",
                fontsize=16, ha="center"
            )

        # Tracer le terrain
        ax.scatter(0, 0, c="pink")
        ax.scatter([10, 0, 10], [0, 20, 20], c="teal")

        ax.hlines([0, 20, 10, 3.05, 16.95], 0, 10, colors="k")
        ax.vlines([0, 10], 0, 20, colors="k")
        ax.vlines(5, 3.05, 16.95, colors="k")

        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(20.5, -0.5)
        ax.set_aspect("equal")

        fig.savefig(f"plot/{f:g}.png", bbox_inches="tight")
        plt.close(fig)
