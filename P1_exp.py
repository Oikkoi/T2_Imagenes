from P1 import obtener_imagen_final
import numpy as np


def exp_1():
    lista_sigmas = np.linspace(0, 10, 41)
    for sigma in lista_sigmas:
        obtener_imagen_final(676767, sigma, False, False, "Exp1")
