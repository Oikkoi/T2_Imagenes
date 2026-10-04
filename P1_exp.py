from P1 import obtener_imagen_final
import numpy as np

semillas = (676767, 64, 69, 415, 511, 1364, 14209)


def exp_1(semilla):
    lista_sigmas = np.linspace(0, 10, 41)
    for sigma in lista_sigmas:
        obtener_imagen_final(semilla, sigma, False, False, "Exp12")


for seed in semillas:
    exp_1(seed)
