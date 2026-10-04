import os
import numpy as np
import matplotlib.pyplot as plt

ruta_csv = os.path.join(os.path.dirname(__file__), "Resultados_P1", "Exp12")
conjunto_seeds = ("676767", "64")

for seed in conjunto_seeds:
    text = f"pruebas_seed_{seed}.csv"
    ruta = os.path.join(ruta_csv, text)
    datos = np.loadtxt(ruta, delimiter=",", skiprows=1)
    sigma = datos[:, 0]
    rmse_circulo = datos[:, 1]
    plt.plot(sigma, rmse_circulo, label="círculo")
    i = np.argmin(rmse_circulo)
    plt.scatter(sigma[i], rmse_circulo[i], marker="o")
    plt.show()
