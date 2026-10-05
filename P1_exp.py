from P1 import obtener_imagen_final
from P1_graficos import graficar_rmse
import numpy as np
import os


def exp_12(semilla, pasada_fina=False):
    seed = str(semilla)
    nombre_archivo = f"pruebas_seed_{seed}.csv"

    lista_sigmas = np.linspace(0, 10, 41)
    if pasada_fina:
        lista_sigmas = np.linspace(1, 2, 101)
    lista_sigmas = np.round(lista_sigmas, 3)
    for sigma in lista_sigmas:
        ruta_guardado = obtener_imagen_final(semilla, sigma, False,
                                             False, "", pasada_fina)

    ruta_csv = os.path.join(ruta_guardado, nombre_archivo)
    nombre_resultado = f"seed_{seed}"
    if pasada_fina:
        nombre_resultado += "_pasada_fina"
    nombre_resultado += ".png"
    ruta_resultado = os.path.join(os.path.dirname(
        __file__), "Resultados_P1", "Exp12", nombre_resultado)
    graficar_rmse(ruta_csv, ruta_resultado, semilla, 40, pasada_fina)


if __name__ == "__main__":
    semillas = (676767, 64, 69, 415, 511, 1364, 14209, 69420)
    for seed in semillas:
        exp_12(seed, False)
        exp_12(seed, True)
