import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io
import os
import csv


def sumar_fondo():
    lienzo = np.zeros((256, 256))
    Y, X = np.ogrid[0:256, 0:256]
    mascara_fondo = (X < 64) | (Y < 64) | (X >= 128+64) | (Y >= 64+128)
    lienzo[mascara_fondo] = 0.15
    return lienzo, mascara_fondo


def sumar_cuadrado():
    lienzo, mascara_fondo = sumar_fondo()
    fondo = lienzo.copy()
    # cuadrado = np.zeros((256, 256))
    # cuadrado[64:(64+128), 64:(64+128)] = 0.3
    Y, X = np.ogrid[0:256, 0:256]
    mascara_x = (64 <= X) & (X < 64 + 128)
    mascara_y = (64 <= Y) & (Y < 64 + 128)
    mascara_cuadrado = mascara_x & mascara_y & (
        (X - 128)**2 + (Y - 128)**2 > 32**2)
    lienzo[mascara_cuadrado] = 0.45
    return lienzo, mascara_cuadrado, mascara_fondo


def crear_lienzo():
    lienzo, mascara_cuadrado, mascara_fondo = sumar_cuadrado()
    Y, X = np.ogrid[0:256, 0:256]
    mascara_circulo = (X - 128)**2 + (Y - 128)**2 <= 32**2
    lienzo[mascara_circulo] = 0.8
    return lienzo, (mascara_circulo, mascara_cuadrado, mascara_fondo)


def añadir_ruido(lienzo, semilla: int):
    np.random.seed(semilla)
    N = 40
    lienzo_poisson = np.random.poisson(N * lienzo)/N
    return lienzo_poisson


def calcular_kernel(sigma):
    if sigma < 0:
        raise ValueError("La desviación estándar no puede ser negativa")
    if sigma == 0:
        lienzo = np.zeros((1, 1)) + 1
        return lienzo
    # 3 sigma porque contiene el 99,46% de la campana, con la función ceil es un poco más
    radio = np.ceil(3*sigma)
    vector_k = np.arange(-radio, radio + 1)
    kernel_1d = np.exp(-(vector_k**2) / (2 * (sigma**2)))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    suma_total = np.sum(kernel_2d)
    kernel_normalizado = kernel_2d/suma_total
    return kernel_normalizado


def aplicar_kernel(imagen, kernel):
    imagen_ruido = cv2.filter2D(imagen, -1, kernel)
    return imagen_ruido


def obtener_imagen_final(semilla: int, desviacion_estandar: float,
                         mostrar_og: bool = True, mostrar_final: bool = True, ruta_complementaria: str = "", pasada_fina=False):
    # Setting para guardar
    ruta = os.path.dirname(__file__)
    ruta_guardado = os.path.join(
        ruta, "Resultados_P1", "Exp12", ruta_complementaria, f"seed_{semilla}")
    if pasada_fina:
        ruta_guardado += "_pasada_fina"
    os.makedirs(ruta_guardado, exist_ok=True)
    print(f"Definida la ruta de guardado: {ruta_guardado}")

    # Código para calcular
    imagen_og, mascaras = crear_lienzo()
    circulo, cuadrada, fondo = mascaras

    if mostrar_og:  # Mostrar imagen original
        visualizacion(imagen_og, "imagen original")

    # Añadir poisson y convolución con el kernel
    imagen_poisson = añadir_ruido(imagen_og, semilla)
    kernel = calcular_kernel(desviacion_estandar)
    imagen_filtrada = aplicar_kernel(imagen_poisson, kernel)

    if mostrar_final:  # Mostrar imagen final
        visualizacion(imagen_filtrada,
                      "imagen con filtro poisson y distribución gaussiana")

    # Segmento de guardado
    ruta_img_og = os.path.join(ruta_guardado, "img_og.png")
    imagen_para_guardar_inicial = (
        imagen_og * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(ruta_img_og, imagen_para_guardar_inicial)
    imagen_para_guardar_final = (
        imagen_filtrada * 255).clip(0, 255).astype(np.uint8)
    nombre_img_filtrada = f"{desviacion_estandar}_img_filtrada.png"
    ruta_img_filtrada = os.path.join(ruta_guardado, nombre_img_filtrada)
    cv2.imwrite(ruta_img_filtrada, imagen_para_guardar_final)

    # Segmento RMSE
    rmse_c = calcular_rmse(imagen_og, imagen_filtrada, circulo)
    rmse_s = calcular_rmse(imagen_og, imagen_filtrada, cuadrada)
    rmse_f = calcular_rmse(imagen_og, imagen_filtrada, fondo)

    mascara_total = circulo | cuadrada | fondo
    rmse_tot = calcular_rmse(imagen_og, imagen_filtrada, mascara_total)

    nombre_csv = f"pruebas_seed_{semilla}.csv"
    ruta_csv = os.path.join(ruta_guardado, nombre_csv)
    existe = os.path.isfile(ruta_csv)

    with open(ruta_csv, mode="a", newline='') as archivo:
        escritor = csv.writer(archivo)
        if not existe:
            escritor.writerow(
                ["sigma", "RMSE circulo", "RMSE cuadrado", "RMSE fondo", "RMSE Imagen total"])
        escritor.writerow([desviacion_estandar, rmse_c,
                          rmse_s, rmse_f, rmse_tot])
    print(
        f"Guardadas las imágenes y RMSE para {semilla}, {desviacion_estandar}")
    return ruta_guardado


def calcular_rmse(imagen_original, imagen_filtrada, mascara):
    pixeles_original = imagen_original[mascara]
    pixeles_filtrada = imagen_filtrada[mascara]
    error = pixeles_original - pixeles_filtrada
    media = np.mean(error**2)
    rmse = np.sqrt(media)
    return rmse


def visualizacion(lienzo, titulo=""):
    cv2.imshow(titulo, lienzo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Sección de valores alterables
    semilla = 676767
    # 0 para la identidad (la imagen final solo tendrá el ruido poisson sin distribución Gaussiana)
    desviacion_estandar = 0

    obtener_imagen_final(semilla, desviacion_estandar)
