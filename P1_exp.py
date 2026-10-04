from P1 import obtener_imagen_final
from P1_graficos import graficar_rmse
import numpy as np


def exp_12(semilla, pasada_fina=False):
    lista_sigmas = np.linspace(0, 10, 41)
    if pasada_fina:
        lista_sigmas = np.linspace(1, 2, 101)
        lista_sigmas = np.round(lista_sigmas, 2)
    for sigma in lista_sigmas:
        ruta_guardado = obtener_imagen_final(semilla, sigma, False,
                                             False, "", pasada_fina)


if __name__ == "__main__":
    semillas = (676767, 64, 69, 415, 511, 1364, 14209)
    for seed in semillas:
        exp_12(seed, False)
        exp_12(seed, True)
