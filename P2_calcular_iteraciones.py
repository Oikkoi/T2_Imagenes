import os
import contextlib
import warnings
from P2 import (cVT, c_p1, c_p2, cargar_imagen_normalizar, agregar_ruido_gaussiano,
                difusion_ansitropica, calcular_rmse, guardar_fila)

seeds = (67676767, 69420, 1234)
imagen = "cameraman.png"
# paso_temporal = fracción * paso_temporal_max, paso_temporal_max = 1/(4*c_max) | c_max = 1/e en TV y 1 en c_p1 y c_p2).

# fracciones para calcular el paso temporal actual
fracciones = (0.1, 0.5, 0.9, 1.4)
es = (0.2, 1, 0.01, 3)  # Valores para e en TV
umbrales_de_contrastes = (0.02, 0.05, 1)
umbrales_laplaciano = (1, 3, 0.1, 5)
pesos_laplaciano = (0.1, 0.125, 0.25)
Ns = (10, 40, 70, 100, 200)  # numeros de corrida
funciones_c = (cVT, c_p1, c_p2)


def es_repetida(funcion_c, e, umbral_de_contraste, umbral_laplaciano, peso_laplaciano):
    """True si la corrida repetiría otra idéntica: cada función ignora algunos parámetros,
    así que solo se corre el primer valor de los que ignora.
    cVT usa solo e | c_p1 usa k y peso | c_p2 usa k, rho y peso"""
    if funcion_c is cVT:
        return (umbral_de_contraste != umbrales_de_contrastes[0]
                or umbral_laplaciano != umbrales_laplaciano[0]
                or peso_laplaciano != pesos_laplaciano[0])
    if funcion_c is c_p1:
        return e != es[0] or umbral_laplaciano != umbrales_laplaciano[0]
    return e != es[0]   # c_p2


# corridas por semilla (para mostrar el avance)
n_p, n_e, n_k, n_r, n_w, n_N = (len(fracciones), len(es), len(umbrales_de_contrastes),
                                len(umbrales_laplaciano), len(pesos_laplaciano), len(Ns))
por_semilla = n_p * n_N * (n_e + n_k * n_w + n_k * n_r * n_w)
total = por_semilla * len(seeds)
hechas = 0

for seed in seeds:
    folder = os.path.join(os.path.dirname(__file__),
                          "Resultados_P2", f"seed_{str(seed)}", f"{imagen.split('.')[0]}")
    os.makedirs(folder, exist_ok=True)
    ruta_csv = os.path.join(folder, "resultados.csv")

    imagen_normalizada = cargar_imagen_normalizar(imagen)
    imagen_ruidosa = agregar_ruido_gaussiano(imagen_normalizada, seed)
    rmse_ruidosa = calcular_rmse(imagen_normalizada, imagen_ruidosa)

    for fraccion in fracciones:
        for e in es:
            for umbral_de_contraste in umbrales_de_contrastes:
                for umbral_laplaciano in umbrales_laplaciano:
                    for peso_laplaciano in pesos_laplaciano:
                        for N in Ns:
                            for funcion_c in funciones_c:
                                if es_repetida(funcion_c, e, umbral_de_contraste,
                                               umbral_laplaciano, peso_laplaciano):
                                    continue

                                if funcion_c is cVT:
                                    c_max = 1 / e
                                else:
                                    c_max = 1.0

                                paso_temporal = fraccion / (4 * c_max)

                                imagen_difusa, c_inicial, c_final = difusion_ansitropica(
                                    imagen_ruidosa, paso_temporal, e, umbral_de_contraste,
                                    umbral_laplaciano, peso_laplaciano, funcion_c, N)

                                rmse_difusa = calcular_rmse(
                                    imagen_normalizada, imagen_difusa)

                                guardar_fila(ruta_csv, "difusa", rmse_difusa, imagen, seed, funcion_c,
                                             paso_temporal, N, e, umbral_de_contraste,
                                             umbral_laplaciano, peso_laplaciano)
                                guardar_fila(ruta_csv, "ruidosa", rmse_ruidosa, imagen, seed, funcion_c,
                                             paso_temporal, N, e, umbral_de_contraste,
                                             umbral_laplaciano, peso_laplaciano)

                                hechas += 1
                                print(f"{hechas}/{total}  seed={seed} {funcion_c.__name__} "
                                      f"paso={paso_temporal} N={N}", flush=True)
