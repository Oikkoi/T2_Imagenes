import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io
import os
import csv

# El siguiente código es para la sección de implementación 1-3


def sumar_fondo() -> (np.array, np.array):
    """
    Crea el lienzo vacío y la máscara booleana con el fondo
    """
    lienzo = np.zeros((256, 256))
    Y, X = np.ogrid[0:256, 0:256]
    mascara_fondo = (X < 64) | (Y < 64) | (X >= 128+64) | (Y >= 64+128)
    lienzo[mascara_fondo] = 0.15
    return lienzo, mascara_fondo


def sumar_cuadrado() -> (np.array, np.array, np.array):
    """
    Llama a sumar_fondo(). Retorna el lienzo (con el fondo y el cuadrado) 
    y las máscaras booleanas del cuadrado y el fondo
    """
    lienzo, mascara_fondo = sumar_fondo()
    fondo = lienzo.copy()
    Y, X = np.ogrid[0:256, 0:256]
    mascara_x = (64 <= X) & (X < 64 + 128)
    mascara_y = (64 <= Y) & (Y < 64 + 128)
    mascara_cuadrado = mascara_x & mascara_y & (
        (X - 128)**2 + (Y - 128)**2 > 32**2)
    lienzo[mascara_cuadrado] = 0.45
    return lienzo, mascara_cuadrado, mascara_fondo


def crear_lienzo() -> (np.array, (np.array, np.array, np.array)):
    """
    Llama a sumar_cuadrado(). Retorna la imagen pedida en el enunciado 
    y una tupla con las máscaras booleanas del círculo, cuadrado y fondo.
    """
    lienzo, mascara_cuadrado, mascara_fondo = sumar_cuadrado()
    Y, X = np.ogrid[0:256, 0:256]
    mascara_circulo = (X - 128)**2 + (Y - 128)**2 <= 32**2
    lienzo[mascara_circulo] = 0.8
    return lienzo, (mascara_circulo, mascara_cuadrado, mascara_fondo)


def añadir_ruido(lienzo, semilla: int) -> np.array:
    """
    Recibe la imagen original y la semilla para añadir el ruido Poisson.
    Retorna la imagen ruidosa.
    """
    np.random.seed(semilla)
    N = 40
    lienzo_poisson = np.random.poisson(N * lienzo)/N
    return lienzo_poisson


def calcular_kernel(sigma) -> np.array:
    """
    Recibe la desviación estándar (no negativa) deseada para el kernel.
    Retorna el kernel normalizado. Si la desviación estándar es 0, retorna la identidad
    """
    if sigma < 0:
        raise ValueError("La desviación estándar no puede ser negativa")
    if sigma == 0:
        lienzo = np.zeros((1, 1)) + 1
        return lienzo
    radio = np.ceil(3*sigma)
    vector_k = np.arange(-radio, radio + 1)
    kernel_1d = np.exp(-(vector_k**2) / (2 * (sigma**2)))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    suma_total = np.sum(kernel_2d)
    kernel_normalizado = kernel_2d/suma_total
    return kernel_normalizado


def aplicar_kernel(imagen, kernel) -> np.array:
    """
    Recibe la imagen con ruido Poisson y aplica el kernel normalizado al convolucionar gracias a cv2.filter2D()
    Retorna la imagen final"""
    imagen_ruido = cv2.filter2D(imagen, -1, kernel)
    return imagen_ruido


def calcular_rmse(imagen_original, imagen_filtrada, mascara) -> float:
    """
    Recibe la imagen original, la imagen final y la máscara en que se desea evaluar.
    Retorna el error entre ambos
    """
    pixeles_original = imagen_original[mascara]
    pixeles_filtrada = imagen_filtrada[mascara]
    error = pixeles_original - pixeles_filtrada
    media = np.mean(error**2)
    rmse = np.sqrt(media)
    return rmse


def obtener_imagen_final(semilla: int, desviacion_estandar: float,
                         mostrar_og: bool = True, mostrar_final: bool = True, ruta_complementaria: str = "", pasada_fina=False):
    """
    Recibe la semilla, la desviación estándar, mostrar resultados intermedios, ruta complementaria para guardar, y el booleando 'pasada_fina' para extraer los mínimos de cada RMSE
    Guarda las imágenes finales y los .csv con los datos en carpetas dedicadas
    Retorna la ruta de guardado
    """
    # Setting para guardar
    ruta = os.path.dirname(__file__)
    ruta_guardado = os.path.join(
        ruta, "Resultados_P1", "Exp12", ruta_complementaria, f"seed_{semilla}")
    if pasada_fina:
        ruta_guardado += "_pasada_fina"
    os.makedirs(ruta_guardado, exist_ok=True)

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

# El siguiente segmento de código es para los apartados 4-6 de implementación:


def calcular_filtro_adaptativo(imagen_filtrada) -> np.array:
    """
    Recibe la imagen filtrada!!!!!

    Gracias a las distinas ejecuciones con distintas semillas, 
    descubrí que el sigma óptimo para las distintas zonas es:
    Cuadrado	1.39
    Círculo	    1.49
    Global	    1.54
    Fondo	    1.76
    """
    xp = [0.15, 0.45, 0.80]
    fp = [1.76, 1.39, 1.49]
    sigma_adaptativo = np.interp(imagen_filtrada, xp, fp)
    return sigma_adaptativo


def banco_sigmas(imagen_poisson, mapa_sigma) -> (np.array, np.array):
    """
    Crea un banco con varias imágenes filtradas para distintos valores tabulados de la
    desgiación estándar. 
    Retorna el banco y los valores de sigma en arrays
    """
    banco_imagenes = []
    cantidad_filtros = 50
    v_min = np.min(mapa_sigma)
    v_max = np.max(mapa_sigma)
    sigmas = np.linspace(v_min, v_max, cantidad_filtros)
    for s in sigmas:
        k = calcular_kernel(s)
        imagen_filtrada = aplicar_kernel(imagen_poisson, k)
        banco_imagenes.append(imagen_filtrada)
    banco_imagenes = np.array(banco_imagenes)
    return banco_imagenes, sigmas


def aplicar_filtro_adaptativo(mapa_sigma, banco_imagenes, sigmas):
    """
    Recibe el mapa de sigma, el banco de imágenes filtradas,
    y la lista uniforme de sigmas del banco. Para cada píxel mezcla linealmente
    las dos imágenes del banco cuyos sigma rodean al sigma de ese píxel.
    Retorna la imagen filtrada adaptativamente.
    """
    if sigmas[-1] == sigmas[0]:  # mapa constante
        return banco_imagenes[0].copy()

    n = len(sigmas)

    posicion = (mapa_sigma - sigmas[0]) / (sigmas[-1] - sigmas[0]) * (n - 1)
    posicion = np.clip(posicion, 0, n - 1)

    abajo = np.minimum(np.floor(posicion).astype(int), n - 2)
    peso = posicion - abajo

    img_abajo = np.take_along_axis(banco_imagenes, abajo[None], axis=0)[0]
    img_arriba = np.take_along_axis(
        banco_imagenes, (abajo + 1)[None], axis=0)[0]
    resultado = (1 - peso) * img_abajo + peso * img_arriba
    return resultado


def mapa_a_imagen(mapa, vmin, vmax, titulo="", escala=3, ancho_barra=30):
    """Convierte un mapa 2D en una imagen BGR con barra de color, escala numérica y título."""
    FUENTE = cv2.FONT_HERSHEY_SIMPLEX
    norm = np.clip((mapa - vmin) / (vmax - vmin), 0, 1)
    color = cv2.applyColorMap(
        (norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    color = cv2.resize(color, None, fx=escala, fy=escala,
                       interpolation=cv2.INTER_NEAREST)
    alto = color.shape[0]

    gradiente = np.linspace(255, 0, alto).astype(np.uint8).reshape(-1, 1)
    barra = cv2.applyColorMap(np.repeat(gradiente, ancho_barra, axis=1),
                              cv2.COLORMAP_VIRIDIS)
    panel = np.full((alto, 80, 3), 255, np.uint8)
    for valor, y in ((vmax, 15), ((vmin + vmax) / 2, alto // 2), (vmin, alto - 5)):
        cv2.putText(panel, f"{valor:.2f}", (5, y), FUENTE, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)

    cuerpo = np.hstack([color, barra, panel])

    cabecera = np.full((30, cuerpo.shape[1], 3), 255, np.uint8)
    cv2.putText(cabecera, titulo, (5, 20), FUENTE,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return np.vstack([cabecera, cuerpo])


# El siguiente segmento de código es para ejecución o visualización general


def visualizacion(lienzo, titulo=""):
    cv2.imshow(titulo, lienzo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Sección de valores alterables
    semilla = 676767
    s = 3  # 0 para la identidad (la imagen final solo tendrá el ruido poisson)
    os.makedirs(os.path.join("Resultados_P1", "Exp12"), exist_ok=True)
    os.makedirs(os.path.join("Resultados_P1", "Exp56"), exist_ok=True)
    imagen_og, mascaras = crear_lienzo()
    circulo, cuadrado, fondo = mascaras
    imagen_poisson = añadir_ruido(imagen_og, semilla)
    k = calcular_kernel(s)
    imagen_filtrada = aplicar_kernel(imagen_poisson, k)
    mapa_sigmas = calcular_filtro_adaptativo(imagen_filtrada)
    banco_imagenes, sigmas = banco_sigmas(imagen_poisson, mapa_sigmas)
    imagen_final = aplicar_filtro_adaptativo(
        mapa_sigmas, banco_imagenes, sigmas)
    imagen_global = aplicar_kernel(imagen_poisson, calcular_kernel(1.54))
    mascara_total = circulo | cuadrado | fondo

    print(
        f"RMSE no adaptativo circulo: {calcular_rmse(imagen_og, imagen_filtrada, circulo)}")
    print(
        f"RMSE no adaptativo cuadrado: {calcular_rmse(imagen_og, imagen_filtrada, cuadrado)}")
    print(
        f"RMSE no adaptativo fondo: {calcular_rmse(imagen_og, imagen_filtrada, fondo)}")
    print(
        f"RMSE no adaptativo global: {calcular_rmse(imagen_og, imagen_filtrada, mascara_total)}")

    print(
        f"RMSE adaptativo circulo: {calcular_rmse(imagen_og, imagen_final, circulo)}")
    print(
        f"RMSE adaptativo cuadrado: {calcular_rmse(imagen_og, imagen_final, cuadrado)}")
    print(
        f"RMSE adaptativo fondo: {calcular_rmse(imagen_og, imagen_final, fondo)}")
    print(
        f"RMSE adaptativo global: {calcular_rmse(imagen_og, imagen_final, mascara_total)}")
    cv2.imwrite(os.path.join("Resultados_P1", "Exp56", "imagen_adaptativa.png"),
                (imagen_final * 255).clip(0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join("Resultados_P1", "Exp56", "imagen_global.png"),
                (imagen_global * 255).clip(0, 255).astype(np.uint8))
    mapa_usado = mapa_a_imagen(mapa_sigmas, mapa_sigmas.min(),
                               mapa_sigmas.max(), "Mapa utilizado")
    diferencia = np.abs(imagen_final - imagen_global)
    print(
        f"diferencia máxima: {np.max(diferencia)}, media: {np.mean(diferencia)}")
    cv2.imwrite(os.path.join("Resultados_P1", "Exp56", "diferencia_x100.png"),
                (diferencia * 100 * 255).clip(0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join("Resultados_P1", "Exp56", "mapa_usado.png"),
                (mapa_usado * 255).clip(0, 255).astype(np.uint8))

    sigma_ideal = calcular_filtro_adaptativo(imagen_og)
    desviacion = np.abs(mapa_sigmas - sigma_ideal)
    print(
        f"desviación máxima: {desviacion.max():.3f}, media: {desviacion.mean():.4f}")
    cv2.imwrite(os.path.join("Resultados_P1", "Exp56", "desviacion_sigma.png"),
                (desviacion / desviacion.max() * 255).astype(np.uint8))
