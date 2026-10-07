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


def cVT(variables: dict):
    """
    coeficiente de difusión propuesto en el enunciado
    retorna la ecuación
    """
    gradiente_u = variables["magnitud_gradiente"]
    e = variables["e"]
    c = 1 / np.sqrt(gradiente_u**2 + e**2)
    return c


def c_p1():
    pass


def c_p2():
    pass


def calcular_rmse(imagen_original, imagen_difusa, ruta_csv) -> float:
    """
    Recibe la imagen original Y la imagen final
    Retorna el error entre ambos
    """
    error = imagen_original - imagen_difusa
    media = np.mean(error**2)
    rmse = np.sqrt(media)

    existe = os.path.isfile(ruta_csv)
    with open(ruta_csv, mode="a", newline='') as archivo:
        escritor = csv.writer(archivo)
        if not existe:
            escritor.writerow(
                ["MSE", "RMSE"])
        escritor.writerow([media,
                          rmse])
    return rmse


def restas_imagen(imagen):
    imagen_pad = np.pad(imagen, 1, mode="edge")
    gradiente_arriba = imagen_pad[:-2, 1:-1] - imagen
    gradiente_abajo = imagen_pad[2:, 1:-1] - imagen
    gradiente_derecha = imagen_pad[1:-1, 2:] - imagen
    gradiente_izquierda = imagen_pad[1:-1, :-2] - imagen
    return gradiente_arriba, gradiente_abajo, gradiente_derecha, gradiente_izquierda


def calcular_laplaciano(dif_arriba, dif_abajo, dif_derecha, dif_izquierda):
    laplaciano = dif_arriba + dif_abajo + dif_derecha + dif_izquierda
    u_x = (dif_derecha - dif_izquierda) / 2
    u_y = (dif_abajo - dif_arriba) / 2
    return laplaciano, u_x, u_y


def difusion_ansitropica(imagen_ruidosa, paso_temporal, e, funcion_c, numero_de_iteraciones: int):
    imagen_actual = imagen_ruidosa.copy()
    for i in range(numero_de_iteraciones):
        print(f"Iniciando iteración {i}")

        arriba, abajo, derecha, izquierda = restas_imagen(imagen_actual)
        laplaciano, u_x, u_y = calcular_laplaciano(
            arriba, abajo, derecha, izquierda)
        magnitud_gradiente = np.sqrt(u_x**2 + u_y**2)

        variables = {
            "laplaciano": laplaciano,
            "gradiente_x": u_x,
            "gradiente_y": u_y,
            "lambda": paso_temporal,
            "e": e,
            "N": numero_de_iteraciones,
            "magnitud_gradiente": magnitud_gradiente
        }
        c = funcion_c(variables)
        print(f"iter {i}: c_min={c.min():.4f}, c_max={c.max():.4f}")
        c_arriba, c_abajo, c_derecha, c_izquierda = restas_imagen(c)

        if (paso_temporal > (1/(4*c.max()))):
            warnings.warn(
                f"Iteración {i}: paso temporal λ={paso_temporal} excede la cota de estabilidad "
                f"λ_max=1/(4·c_max)={1/(4*c.max()):.5f} (c_max={c.max():.3f}). "
                f"El resultado podría mostrar una amplificación en lugar de suavizado\n"
            )
        flujo = ((c + c_arriba / 2) * arriba +
                 (c + c_abajo / 2) * abajo +
                 (c + c_derecha / 2) * derecha +
                 (c + c_izquierda / 2) * izquierda)
        imagen_actual += flujo * paso_temporal
    return imagen_actual


if __name__ == "__main__":
    # variables
    seed = 67676767
    imagen = "cameraman.png"
    paso_temporal = 0.002
    e = 0.01
    N = 40
    nombre_imagen = f"procesada_paso_{str(paso_temporal)}_e_{str(e)}_N_{str(N)}.tiff"
    nombre_ruido = f"ruido.png"
    nombre_csv = f"paso_{str(paso_temporal)}_e_{str(e)}_N_{str(N)}.csv"
    folder = os.path.join(os.path.dirname(__file__),
                          "Resultados_P2", f"seed_{str(seed)}", f"{imagen.split(".")[0]}")
    os.makedirs(folder, exist_ok=True)
    ruta_imagen = os.path.join(folder, nombre_imagen)
    ruta_ruido = os.path.join(folder, nombre_ruido)
    ruta_csv = os.path.join(folder, nombre_csv)

    # test_code
    imagen_normalizada = cargar_imagen_normalizar(imagen)
    imagen_ruidosa = agregar_ruido_gaussiano(imagen_normalizada, seed)
    imagen_difusa = difusion_ansitropica(
        imagen_ruidosa, paso_temporal, e, cVT, N)
    rmse = calcular_rmse(imagen_normalizada, imagen_difusa, ruta_csv)
    imagen_terminada = desnormalizar_imagen(imagen_difusa)
    imagen_ruido = desnormalizar_imagen(imagen_ruidosa)
    io.imsave(ruta_imagen, imagen_terminada)
    io.imsave(ruta_ruido, imagen_ruido)
