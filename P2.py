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


def c_p1(variables: dict):
    """inspirado en el modelo exponencial"""
    u_x = np.abs(variables["u_x_suave"])
    u_y = np.abs(variables["u_y_suave"])
    k = variables["k"]
    E = u_x + u_y
    return np.exp(- ((E/k) ** 2))


def c_p2(variables: dict):
    """inspirado en el modelo fraccional.
    E = (1 - b)·|gradiente u|, con b = clip(r/umbral laplaciano, 0, 1) y r = laplaciano / (gradiente u + δ).
    """
    gradiente_u = variables["magnitud_gradiente_suave"]
    laplaciano = np.abs(variables["laplaciano_suave"])
    k = variables["k"]
    umbral_laplaciano = variables["umbral_laplaciano"]
    razon = laplaciano / (gradiente_u + 1e-6)   # δ=1e-6 evita 0/0
    b = np.clip(razon / umbral_laplaciano, 0, 1)
    E = (1 - b) * gradiente_u
    return 1 / (1 + (E / k) ** 2)


def calcular_rmse(imagen_original, imagen_comparada) -> float:
    """Recibe la imagen original y la imagen a comparar, ambas float en [0,1].
    Retorna el RMSE."""
    error = imagen_original - imagen_comparada
    return float(np.sqrt(np.mean(error ** 2)))


def guardar_fila(ruta_csv, tipo, rmse, imagen, seed, funcion_c,
                 paso_temporal, N, e, k, rho, peso):
    """Agrega una fila al CSV. Los parámetros que la función no usa quedan en '-'."""
    COLUMNAS = ["imagen", "semilla", "funcion", "lambda", "N", "e", "k", "rho",
                "peso_suavizado", "tipo", "MSE", "RMSE"]
    nombre = funcion_c.__name__
    fila = [imagen, seed, nombre, paso_temporal, N,
            e if nombre == "cVT" else "-",
            k if nombre in ("c_p1", "c_p2") else "-",
            rho if nombre == "c_p2" else "-",
            peso if nombre in ("c_p1", "c_p2") else "-",
            tipo, rmse ** 2, rmse]
    existe = os.path.isfile(ruta_csv)
    with open(ruta_csv, mode="a", newline="") as archivo:
        escritor = csv.writer(archivo)
        if not existe:
            escritor.writerow(COLUMNAS)
        escritor.writerow(fila)


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


def difusion_ansitropica(imagen_ruidosa, paso_temporal, e, umbral_de_contraste, umbral_laplaciano, peso_laplaciano: float, funcion_c, numero_de_iteraciones: int):
    if not (0 <= peso_laplaciano <= 0.25):
        raise ValueError("peso_laplaciano debe estar entre 0 y 0.25")

    imagen_actual = imagen_ruidosa.copy()
    for i in range(numero_de_iteraciones):
        print(f"Iniciando iteración {i}")

        arriba, abajo, derecha, izquierda = restas_imagen(imagen_actual)
        laplaciano, u_x, u_y = calcular_laplaciano(
            arriba, abajo, derecha, izquierda)
        magnitud_gradiente = np.sqrt(u_x**2 + u_y**2)

        imagen_suavizada = imagen_actual + peso_laplaciano * laplaciano
        arriba_suave, abajo_suave, derecha_suave, izquierda_suave = restas_imagen(
            imagen_suavizada)
        laplaciano_suave, ux_suave, uy_suave = calcular_laplaciano(
            arriba_suave, abajo_suave, derecha_suave, izquierda_suave)
        gradiente_suavizado = np.sqrt(ux_suave**2 + uy_suave**2)

        variables = {
            "laplaciano_suave": laplaciano_suave,
            "u_x_suave": ux_suave,
            "u_y_suave": uy_suave,
            "laplaciano": laplaciano,
            "u_x": u_x,
            "u_y": u_y,
            "lambda": paso_temporal,
            "e": e,
            "umbral_laplaciano": umbral_laplaciano,
            "k": umbral_de_contraste,
            "N": numero_de_iteraciones,
            "magnitud_gradiente": magnitud_gradiente,
            "magnitud_gradiente_suave": gradiente_suavizado
        }
        c = funcion_c(variables)
        if i == 0:
            mapa_c_inicial = c.copy()
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
    return imagen_actual, mapa_c_inicial, c


if __name__ == "__main__":
    # variables
    seed = 67676767
    imagen = "cameraman.png"
    paso_temporal = 0.125
    e = 0.2
    umbral_de_contraste = 0.02
    umbral_laplaciano = 3
    peso_laplaciano = 0.25
    N = 60
    funcion_c = c_p1

    # zona_nombres
    nombre_imagen = f"procesada_funcion_{(funcion_c.__name__)}_paso_{str(paso_temporal)}_e_{str(e)}_N_{str(N)}_K_{str(umbral_de_contraste)}.tiff"
    nombre_ruido = f"ruido.png"

    folder = os.path.join(os.path.dirname(__file__),
                          "Resultados_P2", f"seed_{str(seed)}", f"{imagen.split(".")[0]}")
    os.makedirs(folder, exist_ok=True)
    ruta_imagen = os.path.join(folder, nombre_imagen)
    ruta_ruido = os.path.join(folder, nombre_ruido)
    ruta_csv = os.path.join(folder, "resultados.csv")

    # test_code
    imagen_normalizada = cargar_imagen_normalizar(imagen)
    imagen_ruidosa = agregar_ruido_gaussiano(imagen_normalizada, seed)
    imagen_difusa, c_inicial, c_final = difusion_ansitropica(
        imagen_ruidosa, paso_temporal, e, umbral_de_contraste, umbral_laplaciano, peso_laplaciano, funcion_c, N)
    imagen_terminada = desnormalizar_imagen(imagen_difusa)
    imagen_ruido = desnormalizar_imagen(imagen_ruidosa)

    # zona guardar
    io.imsave(ruta_imagen, imagen_terminada)
    io.imsave(ruta_ruido, imagen_ruido)
    np.save(os.path.join(
        folder, f"c_inicial_{funcion_c.__name__}_paso_{str(paso_temporal)}_e_{str(e)}_N_{str(N)}_K_{str(umbral_de_contraste)}.npy"), c_inicial)
    np.save(os.path.join(
        folder, f"c_final_{funcion_c.__name__}_paso_{str(paso_temporal)}_e_{str(e)}_N_{str(N)}_K_{str(umbral_de_contraste)}.npy"), c_final)
    rmse_difusa = calcular_rmse(imagen_normalizada, imagen_difusa)
    rmse_ruidosa = calcular_rmse(imagen_normalizada, imagen_ruidosa)
    guardar_fila(ruta_csv, "difusa", rmse_difusa, imagen, seed, funcion_c,
                 paso_temporal, N, e, umbral_de_contraste, umbral_laplaciano, peso_laplaciano)
    guardar_fila(ruta_csv, "ruidosa", rmse_ruidosa, imagen, seed, funcion_c,
                 paso_temporal, N, e, umbral_de_contraste, umbral_laplaciano, peso_laplaciano)
