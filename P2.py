import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io
import os
import csv
import warnings


def visualizacion(imagen, titulo=""):
    cv2.imshow(titulo, imagen)
    while True:
        tecla = cv2.waitKey(0) & 0xFF
        if tecla == 27:
            break
    cv2.destroyAllWindows()


def desnormalizar_imagen(imagen_normalizada: np.array) -> np.array:
    """Recibe una imagen normalizada y retorna una imagen visualizable en escala 0-255"""
    imagen_desnormalizada = imagen_normalizada * 255.0
    imagen_desnormalizada = np.clip(imagen_desnormalizada, a_min=0, a_max=255)
    return imagen_desnormalizada.astype(np.uint8)


def cargar_imagen_normalizar(path_imagen: str) -> np.array:
    """
    Recibe el nombre de una imagen gray y retorna el array normalizado que la representa
    """
    imagen_gray = io.imread(path_imagen, as_gray=True)
    imagen_normalizada = imagen_gray / np.max(imagen_gray)
    imagen_normalizada = np.clip(imagen_normalizada, a_min=0, a_max=1)
    return imagen_normalizada


def agregar_ruido_gaussiano(imagen: np.array, seed: int) -> np.array:
    """
    Recibe la imagen normalizada y la semilla de generación de ruido,
    retorna la imagen con ruido gaussiano
    """
    np.random.seed(seed)
    ruido_gaussiano = np.random.normal(loc=0, scale=0.05, size=imagen.shape)
    imagen_ruidosa = np.clip(imagen + ruido_gaussiano, a_min=0, a_max=1)
    return imagen_ruidosa


def coeficiente_de_difusion(e: float, gradiente_u: np.array):
    """
    coeficiente de difusión, función de E(para cTV E=épsilon) y el gradiente de los cuatro vecinos
    retorna la ecuación
    """
    # c = e
    c = 1 / np.sqrt(gradiente_u**2 + e**2)
    return c


def calcular_rmse(imagen_original, imagen_difusa, file_name) -> float:
    """
    Recibe la imagen original Y la imagen final
    Retorna el error entre ambos
    """
    error = imagen_original - imagen_difusa
    media = np.mean(error**2)
    rmse = np.sqrt(media)

    return rmse


def difusion_ansitropica(imagen_ruidosa, paso_temporal, e, numero_de_iteraciones: int):
    imagen_actual = imagen_ruidosa.copy()
    c_max_total = 0
    for i in range(numero_de_iteraciones):
        print(f"Iniciando iteración {i}")
        imagen_pad = np.pad(imagen_actual, 1, mode="edge")
        gradiente_arriba = imagen_pad[:-2, 1:-1] - imagen_actual
        gradiente_abajo = imagen_pad[2:, 1:-1] - imagen_actual
        gradiente_derecha = imagen_pad[1:-1, 2:] - imagen_actual
        gradiente_izquierda = imagen_pad[1:-1, :-2] - imagen_actual

        u_x = (gradiente_derecha - gradiente_izquierda) / 2
        u_y = (gradiente_arriba - gradiente_abajo) / 2
        magnitud_gradiente = np.sqrt(u_x**2 + u_y**2)

        c = coeficiente_de_difusion(e, magnitud_gradiente)
        print(f"iter {i}: c_min={c.min():.4f}, c_max={c.max():.4f}")
        if (paso_temporal > (1/(4*c.max()))):
            warnings.warn(
                f"Iteración {i}: paso temporal λ={paso_temporal} excede la cota de estabilidad "
                f"λ_max=1/(4·c_max)={1/(4*c.max()):.5f} (c_max={c.max():.3f}). "
                f"El resultado podría mostrar una amplificación en lugar de suavizado"
            )
        flujo = (gradiente_arriba + gradiente_abajo +
                 gradiente_derecha + gradiente_izquierda) * c
        imagen_actual += flujo * paso_temporal

    return imagen_actual


if __name__ == "__main__":

    # variables
    seed = 67676767
    imagen = "imagen_gray.jpg"
    paso_temporal = 0.002
    e = 0.01
    N = 40
    nombre_imagen = f"{imagen.split(".")[0]}_procesada_paso_{str(paso_temporal)}_e_{str(e)}_N_{str(N)}.tiff"
    folder = os.path.join(os.path.dirname(__file__),
                          "Resultados_P2", "test")
    os.makedirs(folder, exist_ok=True)
    nombre_completo = folder + nombre_imagen

    # test_code
    imagen_normalizada = cargar_imagen_normalizar(imagen)
    imagen_ruidosa = agregar_ruido_gaussiano(imagen_normalizada, seed)
    imagen_difusa = difusion_ansitropica(imagen_ruidosa, paso_temporal, e, N)
    imagen_terminada = desnormalizar_imagen(imagen_difusa)
    try:
        io.imsave(nombre_completo, imagen_terminada)
    except:
        visualizacion(imagen_terminada, "test")
