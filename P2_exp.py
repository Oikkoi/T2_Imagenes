import os
import matplotlib.pyplot as plt
from P2 import difusion_ansitropica, calcular_rmse, cVT, cargar_imagen_normalizar, desnormalizar_imagen, agregar_ruido_gaussiano, c_p1, c_p2

folder = os.path.join(os.path.dirname(__file__), "Resultados_P2")


# mapas cTV para distintos e
def mapas_cTV(es_mapas: tuple = (0.01, 0.2, 1, 3)):
    # Parámetros
    seed = 67676767
    imagen = "cameraman.png"
    paso_temporal = 0.002
    umbral_de_contraste = 0.02
    umbral_laplaciano = 1
    peso_laplaciano = 0.25
    e = 0.1
    N = 10
    fraccion = 0.4
    funcion_c = cVT

    # Llamado funciones
    imagen_normalizada = cargar_imagen_normalizar(imagen)
    imagen_ruidosa = agregar_ruido_gaussiano(imagen_normalizada, seed)
    imagen_difusa, c_inicial, c_final = difusion_ansitropica(
        imagen_ruidosa, paso_temporal, e, umbral_de_contraste, umbral_laplaciano, peso_laplaciano, funcion_c, N)
    imagen_terminada = desnormalizar_imagen(imagen_difusa)
    imagen_ruido = desnormalizar_imagen(imagen_ruidosa)
    # Iteracion
    ruta_mapas = os.path.join(folder, "mapas_cTV.png")
    fig, ax = plt.subplots(3, len(es_mapas), figsize=(16, 12))
    for j, e_prueba in enumerate(es_mapas):
        c_max = 1 / e_prueba
        imagen_prueba, c_ini, c_fin = difusion_ansitropica(
            imagen_ruidosa, paso_temporal, e_prueba, umbral_de_contraste,
            umbral_laplaciano, peso_laplaciano, cVT, N)
        rmse_prueba = calcular_rmse(imagen_normalizada, imagen_prueba)

        mapa = ax[0, j].imshow(
            e_prueba * c_ini, cmap="viridis", vmin=0, vmax=1)
        ax[0, j].set_title(f"c inicial | e={e_prueba}, λ={paso_temporal:.4g}")
        ax[1, j].imshow(e_prueba * c_fin, cmap="viridis", vmin=0, vmax=1)
        ax[1, j].set_title(f"c final | N={N}")
        ax[2, j].imshow(imagen_prueba, cmap="gray", vmin=0, vmax=1)
        ax[2, j].set_title(f"resultado | RMSE={rmse_prueba:.4f}")

    for eje in ax.ravel():
        eje.axis("off")
    fig.colorbar(mapa, ax=ax[:2, :], shrink=0.6, label="e · c_TV")
    fig.savefig(ruta_mapas, dpi=150, bbox_inches="tight")
    return


mapas_cTV()


def mapas_cTV_lambda(es_mapas: tuple = (0.01, 0.2, 1, 3)):
    # Parámetros
    seed = 67676767
    imagen = "cameraman.png"
    pasos_temporales = (0.01, 0.05, 0.1, 1)
    umbral_de_contraste = 0.02
    umbral_laplaciano = 1
    peso_laplaciano = 0.25
    e = 0.1
    N = 10
    fraccion = 0.4
    funcion_c = cVT
    for paso_temporal in pasos_temporales:
        # Llamado funciones
        imagen_normalizada = cargar_imagen_normalizar(imagen)
        imagen_ruidosa = agregar_ruido_gaussiano(imagen_normalizada, seed)
        imagen_difusa, c_inicial, c_final = difusion_ansitropica(
            imagen_ruidosa, paso_temporal, e, umbral_de_contraste, umbral_laplaciano, peso_laplaciano, funcion_c, N)
        imagen_terminada = desnormalizar_imagen(imagen_difusa)
        imagen_ruido = desnormalizar_imagen(imagen_ruidosa)
        # Iteracion
        ruta_mapas = os.path.join(
            folder, f"mapas_cTV_lambda_{paso_temporal}.png")
        fig, ax = plt.subplots(3, len(es_mapas), figsize=(16, 12))

        for j, e_prueba in enumerate(es_mapas):
            c_max = 1 / e_prueba
            imagen_prueba, c_ini, c_fin = difusion_ansitropica(
                imagen_ruidosa, paso_temporal, e_prueba, umbral_de_contraste,
                umbral_laplaciano, peso_laplaciano, cVT, N)
            rmse_prueba = calcular_rmse(imagen_normalizada, imagen_prueba)

            mapa = ax[0, j].imshow(
                e_prueba * c_ini, cmap="viridis", vmin=0, vmax=1)
            ax[0, j].set_title(
                f"c inicial | e={e_prueba}, λ={paso_temporal:.4g}")
            ax[1, j].imshow(e_prueba * c_fin, cmap="viridis", vmin=0, vmax=1)
            ax[1, j].set_title(f"c final | N={N}")
            ax[2, j].imshow(imagen_prueba, cmap="gray", vmin=0, vmax=1)
            ax[2, j].set_title(f"resultado | RMSE={rmse_prueba:.4f}")
        for eje in ax.ravel():
            eje.axis("off")
        fig.colorbar(mapa, ax=ax[:2, :], shrink=0.6, label="e · c_TV")
        fig.savefig(ruta_mapas, dpi=150, bbox_inches="tight")
    return


# mapas_cTV_lambda()


def mapas_cp1_lambda(es_mapas: tuple = (0.01, 0.2, 1, 3)):
    # Parámetros
    seed = 67676767
    imagen = "cameraman.png"
    umbral_laplaciano = 1
    peso_laplaciano = 0.25
    paso_temporal = 0.1
    e = 0.1
    N = 60
    fraccion = 0.4
    funcion_c = c_p1
    for umbral_de_contraste in es_mapas:
        # Llamado funciones
        imagen_normalizada = cargar_imagen_normalizar(imagen)
        imagen_ruidosa = agregar_ruido_gaussiano(imagen_normalizada, seed)
        imagen_difusa, c_inicial, c_final = difusion_ansitropica(
            imagen_ruidosa, paso_temporal, e, umbral_de_contraste, umbral_laplaciano, peso_laplaciano, funcion_c, N)
        imagen_terminada = desnormalizar_imagen(imagen_difusa)
        imagen_ruido = desnormalizar_imagen(imagen_ruidosa)
        # Iteracion
        ruta_mapas = os.path.join(
            folder, f"mapas_cp1_lambda_{paso_temporal}.png")
        fig, ax = plt.subplots(3, len(es_mapas), figsize=(16, 12))

        for j, e_prueba in enumerate(es_mapas):
            c_max = 1 / e_prueba
            imagen_prueba, c_ini, c_fin = difusion_ansitropica(
                imagen_ruidosa, paso_temporal, e_prueba, umbral_de_contraste,
                umbral_laplaciano, peso_laplaciano, c_p1, N)
            rmse_prueba = calcular_rmse(imagen_normalizada, imagen_prueba)

            mapa = ax[0, j].imshow(
                e_prueba * c_ini, cmap="viridis", vmin=0, vmax=1)
            ax[0, j].set_title(
                f"c inicial | k={e_prueba}, λ={paso_temporal:.4g}")
            ax[1, j].imshow(e_prueba * c_fin, cmap="viridis", vmin=0, vmax=1)
            ax[1, j].set_title(f"c final | N={N}")
            ax[2, j].imshow(imagen_prueba, cmap="gray", vmin=0, vmax=1)
            ax[2, j].set_title(f"resultado | RMSE={rmse_prueba:.4f}")
        for eje in ax.ravel():
            eje.axis("off")
        fig.colorbar(mapa, ax=ax[:2, :], shrink=0.6, label="k · c_p1")
        fig.savefig(ruta_mapas, dpi=150, bbox_inches="tight")
    return


def mapas_cp2_lambda(es_mapas: tuple = (0.01, 0.2, 1, 3)):
    # Parámetros
    seed = 67676767
    imagen = "cameraman.png"
    paso_temporal = 0.02
    umbral_laplaciano = 1
    peso_laplaciano = 0.25
    e = 0.1
    N = 30
    fraccion = 0.4
    funcion_c = c_p1
    for umbral_de_contraste in es_mapas:
        # Llamado funciones
        imagen_normalizada = cargar_imagen_normalizar(imagen)
        imagen_ruidosa = agregar_ruido_gaussiano(imagen_normalizada, seed)
        imagen_difusa, c_inicial, c_final = difusion_ansitropica(
            imagen_ruidosa, paso_temporal, e, umbral_de_contraste, umbral_laplaciano, peso_laplaciano, funcion_c, N)
        imagen_terminada = desnormalizar_imagen(imagen_difusa)
        imagen_ruido = desnormalizar_imagen(imagen_ruidosa)
        # Iteracion
        ruta_mapas = os.path.join(
            folder, f"mapas_cp2_lambda_{paso_temporal}.png")
        fig, ax = plt.subplots(3, len(es_mapas), figsize=(16, 12))

        for j, e_prueba in enumerate(es_mapas):
            c_max = 1 / e_prueba
            imagen_prueba, c_ini, c_fin = difusion_ansitropica(
                imagen_ruidosa, paso_temporal, e_prueba, umbral_de_contraste,
                umbral_laplaciano, peso_laplaciano, c_p1, N)
            rmse_prueba = calcular_rmse(imagen_normalizada, imagen_prueba)

            mapa = ax[0, j].imshow(
                e_prueba * c_ini, cmap="viridis", vmin=0, vmax=1)
            ax[0, j].set_title(
                f"c inicial | k={e_prueba}, λ={paso_temporal:.4g}")
            ax[1, j].imshow(e_prueba * c_fin, cmap="viridis", vmin=0, vmax=1)
            ax[1, j].set_title(f"c final | N={N}")
            ax[2, j].imshow(imagen_prueba, cmap="gray", vmin=0, vmax=1)
            ax[2, j].set_title(f"resultado | RMSE={rmse_prueba:.4f}")
        for eje in ax.ravel():
            eje.axis("off")
        fig.colorbar(mapa, ax=ax[:2, :], shrink=0.6, label="k · c_p2")
        fig.savefig(ruta_mapas, dpi=150, bbox_inches="tight")
    return


# mapas_cp1_lambda()
# mapas_cp2_lambda()

def sacar_imagenes():
