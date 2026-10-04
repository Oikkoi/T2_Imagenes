import os
import numpy as np
import matplotlib.pyplot as plt


def graficar_rmse(ruta_csv, ruta_guardado, semilla, N, pasada_fina: bool):
    datos = np.loadtxt(ruta_csv, delimiter=",", skiprows=1)
    datos = datos[np.argsort(datos[:, 0])]
    sigma = datos[:, 0]

    curvas = {
        "Círculo": datos[:, 1],
        "Cuadrado": datos[:, 2],
        "Fondo": datos[:, 3],
        "Global": datos[:, 4],
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for nombre, rmse in curvas.items():
        linea, = ax.plot(sigma, rmse, label=nombre)
        color = linea.get_color()

        ax.axhline(rmse[0], color=color, linestyle=":", alpha=0.5)

        i = np.argmin(rmse)
        ax.scatter(sigma[i], rmse[i], color=color, edgecolor="k", zorder=3,
                   marker="o",
                   s=60,
                   label=f"mín. {nombre}: σ={sigma[i]:.2f}, RMSE={rmse[i]:.4f}")

    ax.set_xlabel("σ del filtro Gaussiano")
    ax.set_ylabel("RMSE")
    ax.set_yscale("linear")
    titulo = f"RMSE(σ) por región | semilla={semilla}, N={N}, σ∈[{sigma[0]:g}, {sigma[-1]:g}], {len(sigma)} valores\n(líneas punteadas = sin Filtro)"
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")  # antes de show


if __name__ == "__main__":
    ruta_parcial = os.path.join(os.path.dirname(
        __file__), "Resultados_P1", "Exp12")
    conjunto_seeds = (676767, 64, 69, 415, 511, 1364, 14209)
    pasadas = (False, True)
    for pasada_fina in pasadas:
        for seed in conjunto_seeds:
            seed = str(seed)
            nombre_archivo = f"pruebas_seed_{seed}.csv"
            if pasada_fina:
                ruta_csv = os.path.join(
                    ruta_parcial, f"seed_{seed}_pasada_fina", nombre_archivo)
                ruta_guardado = os.path.join(
                    ruta_parcial, f"{seed}_pasada_fina.png")
            else:
                ruta_csv = os.path.join(
                    ruta_parcial, f"seed_{seed}", nombre_archivo)
                ruta_guardado = os.path.join(
                    ruta_parcial, f"{seed}.png")
            graficar_rmse(ruta_csv, ruta_guardado, seed, 40, pasada_fina)
