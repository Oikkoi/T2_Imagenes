import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io


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

    # Sección de código
    imagen_og, mascaras = crear_lienzo()
    circulo, cuadrada, fondo = mascaras
    visualizacion(imagen_og, "imagen original")
    lienzo_poisson = añadir_ruido(imagen_og, semilla)
    kernel = calcular_kernel(desviacion_estandar)
    imagen_filtrada = aplicar_kernel(lienzo_poisson, kernel)
    visualizacion(imagen_filtrada,
                  "imagen con filtro poisson y distribución gaussiana")
    print("RMSE circulo (kernel aplicado):", calcular_rmse(
        imagen_og, imagen_filtrada, circulo))
    print("RMSE cuadrado (kernel aplicado):", calcular_rmse(
        imagen_og, imagen_filtrada, cuadrada))
    print("RMSE fondo (kernel aplicado):", calcular_rmse(
        imagen_og, imagen_filtrada, fondo))
