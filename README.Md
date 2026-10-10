# P1 (`P1.py`, `P1_exp.py`, `P1_graficos.py`):
**Para hacer esta pregunta, fueron utilizadas las librerías `numpy`, `matplotlib`, `cv2` (OpenCV), `os` y `csv`.**

## `P1.py`

Se definieron las siguientes funciones:
1. *sumar_fondo*: No recibe argumentos. Crea un lienzo de ceros de 256x256 con `np.zeros` y usa `np.ogrid` para generar los índices X e Y. La máscara booleana del fondo corresponde al marco de 64 píxeles de ancho que rodea la imagen (`X < 64`, `Y < 64`, `X >= 192` o `Y >= 192`), y sobre ella se asigna el valor `0.15`. Retorna el lienzo y la máscara del fondo.

2. *sumar_cuadrado*: Llama a *sumar_fondo* y luego calcula la máscara del cuadrado central (entre 64 y 192 en ambos ejes). A dicha máscara se le **quita** el círculo central (los píxeles a distancia menor o igual a 32 del centro), de modo que el cuadrado queda con un hueco. Asigna `0.45` en la máscara obtenida. Retorna el lienzo y las máscaras del cuadrado y del fondo.

3. *crear_lienzo*: Llama a *sumar_cuadrado*, calcula la máscara del círculo (centro en (128, 128) y radio 32) y le asigna el valor `0.8`. Retorna la imagen pedida en el enunciado y una tupla con las máscaras `(circulo, cuadrado, fondo)`. Las tres máscaras son disjuntas y juntas cubren toda la imagen.

4. *añadir_ruido*: Recibe la imagen original y la semilla. Fija la semilla con `np.random.seed` y usa `np.random.poisson` sobre `N * lienzo` (con `N = 40`), para luego dividir el resultado por `N` y volver a la escala original. Retorna la imagen ruidosa.

5. *calcular_kernel*: Recibe la desviación estándar sigma. Si es negativa, levanta un **ValueError**. Si es 0, retorna un kernel de 1x1 con valor 1 (la identidad). En otro caso, calcula el radio como `np.ceil(3*sigma)`, genera el vector `-radio, ..., radio` con `np.arange`, evalúa la gaussiana 1D, y usa `np.outer` para formar el kernel 2D. Finalmente lo normaliza dividiendo por su suma. Retorna el kernel.

6. *aplicar_kernel*: Recibe la imagen y el kernel. Los convoluciona con `cv2.filter2D`. Retorna la imagen filtrada.

7. *calcular_rmse*: Recibe la imagen original, la imagen filtrada y una máscara booleana. Extrae los píxeles de ambas imágenes bajo la máscara, calcula el error, y retorna la raíz de la media de su cuadrado (usando `np.mean` y `np.sqrt`).

8. *obtener_imagen_final*: Recibe la semilla, la desviación estándar, dos booleanos para mostrar la imagen original y la final, una ruta complementaria y el booleano `pasada_fina`. Crea (con `os.makedirs`) la carpeta `Resultados_P1/Exp12/<ruta_complementaria>/seed_<semilla>` (con el sufijo `_pasada_fina` si corresponde). Luego llama a *crear_lienzo*, *añadir_ruido*, *calcular_kernel* y *aplicar_kernel* en ese orden, y guarda con `cv2.imwrite` la imagen original (`img_og.png`) y la filtrada (`<sigma>_img_filtrada.png`), ambas pasadas a `uint8`. Calcula el RMSE en el círculo, el cuadrado, el fondo y la imagen completa (la unión de las tres máscaras), y agrega una fila al archivo `pruebas_seed_<semilla>.csv` con las columnas `sigma`, `RMSE circulo`, `RMSE cuadrado`, `RMSE fondo` y `RMSE Imagen total` (el encabezado se escribe solo si el archivo no existe). Retorna la ruta de guardado.

9. *calcular_filtro_adaptativo*: Recibe la imagen **filtrada**. A partir de las ejecuciones de `P1_exp.py` con distintas semillas, se obtuvieron los sigma óptimos de cada zona (cuadrado 1.39, círculo 1.49, global 1.54 y fondo 1.76). Usa `np.interp` con las intensidades de cada región (`0.15`, `0.45`, `0.80`) como puntos de referencia y los sigma de fondo, cuadrado y círculo como valores. Así, cada píxel recibe un sigma según su intensidad, con una transición suave entre regiones. Retorna el mapa de sigma.

10. *banco_sigmas*: Recibe la imagen con ruido Poisson y el mapa de sigma. Crea 50 valores de sigma uniformemente espaciados entre el mínimo y el máximo del mapa con `np.linspace`, y para cada uno calcula el kernel y filtra la imagen ruidosa. Retorna el banco de imágenes (como un array de `numpy`) y los valores de sigma.

11. *aplicar_filtro_adaptativo*: Recibe el mapa de sigma, el banco de imágenes y los sigma del banco. Si el mapa es constante, retorna la primera imagen del banco. En otro caso, calcula para cada píxel la posición (fraccionaria) de su sigma dentro del banco, la limita con `np.clip`, y obtiene el índice de la imagen de abajo (con `np.floor`) y el peso de la mezcla. Con `np.take_along_axis` extrae, para cada píxel, el valor de las dos imágenes del banco que rodean su sigma, y las mezcla linealmente. Retorna la imagen filtrada adaptativamente.

12. *mapa_a_imagen*: Recibe un mapa 2D, los valores mínimo y máximo de la escala, el título, el factor de escala y el ancho de la barra. Normaliza el mapa, le aplica el mapa de colores `viridis` con `cv2.applyColorMap`, lo agranda con `cv2.resize` (interpolación por vecino más cercano) y le agrega al lado una barra de color con los valores numéricos (máximo, medio y mínimo) escritos con `cv2.putText`, y arriba una cabecera con el título. Retorna la imagen BGR resultante.

13. *visualizacion*: Recibe una imagen y un título. Muestra la imagen en una ventana con `cv2.imshow` y la cierra al presionar cualquier tecla.

### Uso de `P1.py`:
Se puede ejecutar el archivo directamente.

Al final del archivo (bajo `if __name__ == "__main__":`) se encuentran los parámetros modificables, siendo estos `semilla` y `s`.

`semilla` es el int usado para generar el ruido Poisson. Ejemplo: `676767`.

`s` es el sigma del filtro gaussiano que se usa para obtener la imagen filtrada (la que luego determina el mapa de sigma). Si vale `0`, se usa la identidad y la imagen solo tendrá el ruido Poisson.

Al ejecutarlo se crean las carpetas `Resultados_P1/Exp12` y `Resultados_P1/Exp56`, se imprimen en consola los RMSE (circulo, cuadrado, fondo y global) del filtro no adaptativo y del adaptativo, la diferencia máxima y media entre el filtro adaptativo y el global (sigma = 1.54), y la desviación máxima y media entre el mapa de sigma obtenido desde la imagen filtrada y el mapa ideal (obtenido desde la imagen original). En `Resultados_P1/Exp56` se guardan `imagen_adaptativa.png`, `imagen_global.png`, `diferencia_x100.png` (la diferencia amplificada 100 veces), `mapa_usado.png` y `desviacion_sigma.png`.

## `P1_exp.py`

Se definió la siguiente función:
1. *exp_12*: Recibe la semilla y el booleano `pasada_fina` (por defecto `False`). Define la lista de sigma a probar: 41 valores entre 0 y 10 con `np.linspace` o, si es pasada fina, 101 valores entre 1 y 2. Los redondea a 3 decimales con `np.round` y, para cada uno, llama a *obtener_imagen_final* de `P1.py`, que va agregando una fila al `.csv` de la semilla. Finalmente llama a *graficar_rmse* de `P1_graficos.py` con dicho `.csv` y guarda el gráfico en `Resultados_P1/Exp12` bajo el nombre `seed_<semilla>.png` (o `seed_<semilla>_pasada_fina.png`).

### Uso de `P1_exp.py`:
Se puede ejecutar el archivo directamente. Se modifica la variable `semillas` al final del archivo, que es una tupla con las semillas a probar. Para cada una se hace primero la pasada normal y luego la pasada fina. **Importante:** *obtener_imagen_final* agrega filas al `.csv` sin borrarlo, por lo que si se vuelve a ejecutar con la misma semilla hay que eliminar antes la carpeta correspondiente, o las filas quedarán duplicadas.

## `P1_graficos.py`

Se definió la siguiente función:
1. *graficar_rmse*: Recibe la ruta del `.csv`, la ruta donde guardar el gráfico, la semilla, N y el booleano `pasada_fina`. Lee el `.csv` con `np.loadtxt` (saltando el encabezado) y ordena las filas por sigma con `np.argsort`. Con `matplotlib` grafica las cuatro curvas de RMSE en función de sigma (círculo, cuadrado, fondo y global). Si el sigma 0 está entre los valores probados, agrega una línea punteada horizontal con el RMSE de solo Poisson para cada curva. Para cada curva se calcula el mínimo con `np.argmin` y se marca con un punto, indicando en la leyenda su sigma y su RMSE. Guarda el gráfico con `savefig` (dpi 150) y no lo muestra.

### Uso de `P1_graficos.py`:
Se puede ejecutar el archivo directamente para regenerar los gráficos a partir de los `.csv` ya existentes en `Resultados_P1/Exp12`, sin volver a calcular nada. Se modifica la tupla `conjunto_seeds` al final del archivo. Los gráficos se guardan como `<semilla>.png` y `<semilla>_pasada_fina.png`.


# P2 (`P2.py`, `P2_exp.py`, `P2_calcular_iteraciones.py`):
**Para hacer esta pregunta, fueron utilizadas las librerías `numpy`, `matplotlib`, `cv2` (OpenCV), `skimage`, `os`, `csv` y `warnings`.**

## `P2.py`

Se definieron las siguientes funciones:
1. *visualizacion*: Recibe una imagen y un título. Muestra la imagen con `cv2.imshow` y mantiene la ventana abierta hasta que se presiona la tecla `Esc`.

2. *desnormalizar_imagen*: Recibe una imagen normalizada (entre 0 y 1), la multiplica por 255, la limita entre 0 y 255 con `np.clip` y la convierte a `uint8`. Retorna la imagen visualizable.

3. *cargar_imagen_normalizar*: Recibe el nombre de la imagen. Usa `io.imread` de `skimage` con `as_gray=True` para obtener la imagen en escala de grises, y la divide por su máximo para dejarla entre 0 y 1. Retorna el array normalizado.

4. *agregar_ruido_gaussiano*: Recibe la imagen normalizada y la semilla. Fija la semilla con `np.random.seed` y suma un ruido gaussiano de media 0 y desviación estándar 0.05 (con `np.random.normal`). Luego limita el resultado entre 0 y 1 con `np.clip`. Retorna la imagen ruidosa.

5. *cVT*: Recibe el diccionario de variables. Es el coeficiente de difusión propuesto en el enunciado, `c = 1 / sqrt(|∇u|² + e²)`, donde |∇u| es la magnitud del gradiente de la imagen actual. Retorna el mapa de `c`. Como `c ≤ 1/e`, su máximo teórico es `1/e`.

6. *c_p1*: Recibe el diccionario de variables. Está inspirado en el modelo exponencial: calcula `E = |u_x| + |u_y|` con las derivadas de la imagen **suavizada** y retorna `exp(-(E/k)²)`. Su máximo teórico es 1.

7. *c_p2*: Recibe el diccionario de variables. Está inspirado en el modelo fraccional. Calcula la razón `r = |laplaciano| / (|∇u| + δ)` con la imagen **suavizada** (con `δ = 1e-6` para evitar 0/0), luego `b = clip(r / rho, 0, 1)` y `E = (1 - b)·|∇u|`. Retorna `1 / (1 + (E/k)²)`. La idea es que donde el laplaciano es grande en comparación al gradiente (es decir, donde hay ruido y no un borde) `b` se acerca a 1, `E` se acerca a 0 y `c` a 1, difundiendo más. Su máximo teórico es 1.

8. *calcular_rmse*: Recibe la imagen original y la comparada (ambas entre 0 y 1). Retorna la raíz de la media del error al cuadrado, **redondeada a 5 decimales** con `np.round`.

9. *guardar_fila*: Recibe la ruta del `.csv`, el tipo de fila (`"difusa"` o `"ruidosa"`), el RMSE, la imagen, la semilla, la función c y los parámetros usados. Agrega una fila con las columnas `imagen`, `semilla`, `funcion`, `lambda`, `N`, `e`, `k`, `rho`, `peso_suavizado`, `tipo`, `MSE` y `RMSE` (el encabezado se escribe solo si el archivo no existe). Los parámetros que la función c usada no utiliza quedan en `-`: `cVT` usa solo `e`, `c_p1` usa `k` y `peso_suavizado`, y `c_p2` usa `k`, `rho` y `peso_suavizado`.

10. *restas_imagen*: Recibe una imagen. La extiende un píxel por cada lado con `np.pad` en modo `edge` (para que el borde no genere flujo) y retorna las cuatro diferencias entre cada vecino (arriba, abajo, derecha, izquierda) y el píxel central, usando slicing.

11. *calcular_laplaciano*: Recibe las cuatro diferencias de *restas_imagen*. El laplaciano es la suma de las cuatro. También calcula las derivadas centrales `u_x = (derecha - izquierda)/2` y `u_y = (abajo - arriba)/2`. Retorna el laplaciano, `u_x` y `u_y`.

12. *difusion_ansitropica*: Recibe la imagen ruidosa, el paso temporal, `e`, el umbral de contraste (`k`), el umbral laplaciano (`rho`), el peso laplaciano, la función c y el número de iteraciones. Si el peso laplaciano no está entre 0 y 0.25 (cota de estabilidad de un paso de difusión), levanta un **ValueError**. Luego, en cada iteración:
    - Calcula las diferencias, el laplaciano, las derivadas y la magnitud del gradiente de la imagen actual.
    - Crea una imagen suavizada sumando a la imagen actual el laplaciano multiplicado por el peso laplaciano (un paso de difusión lineal), y calcula sobre ella las mismas cantidades. Esto hace que las funciones `c_p1` y `c_p2` sean menos sensibles al ruido.
    - Junta todo en un diccionario y llama a la función c para obtener el mapa de coeficientes. En la primera iteración guarda una copia de este mapa.
    - Calcula las diferencias del mapa `c` con *restas_imagen* y, si el paso temporal supera la cota `1/(4·c_max)`, lanza un `warnings.warn` indicando que el resultado podría amplificarse en lugar de suavizarse.
    - Calcula el flujo como la suma de las cuatro direcciones de `(c + c_vecino/2) · diferencia` y actualiza la imagen sumando el flujo multiplicado por el paso temporal.

    Retorna la imagen final, el mapa de `c` inicial y el mapa de `c` de la última iteración.

### Uso de `P2.py`:
Se puede ejecutar el archivo directamente. Es necesario tener la imagen (por ejemplo `cameraman.png`) en la misma carpeta que `P2.py`.

Al final del archivo (bajo `if __name__ == "__main__":`) se encuentran los parámetros modificables, siendo estos `seed`, `imagen`, `paso_temporal`, `e`, `umbral_de_contraste`, `umbral_laplaciano`, `peso_laplaciano`, `N` y `funcion_c`.

`seed` es el int de la semilla del ruido gaussiano. Ejemplo: `67676767`.

`imagen` es un string con el nombre del archivo con su extensión. Ejemplo: `"cameraman.png"`.

`paso_temporal` es el paso de la difusión (λ). Para que sea estable debe ser menor o igual a `1/(4·c_max)`.

`e` es el parámetro de `cVT`, `umbral_de_contraste` es el parámetro `k` de `c_p1` y `c_p2`, `umbral_laplaciano` es el parámetro `rho` de `c_p2`, y `peso_laplaciano` es el peso del suavizado previo (float entre 0 y 0.25).

`N` es el int con el número de iteraciones.

`funcion_c` es la función de difusión a usar: `cVT`, `c_p1` o `c_p2`.

Al ejecutarlo se crea la carpeta `Resultados_P2`, donde se guardan la imagen procesada (`.tiff`) y la imagen con ruido (`ruido.png`), y se imprime en consola el RMSE de la imagen difusa. El guardado de los mapas `c_inicial` y `c_final` (`.npy`) y de las filas en `resultados.csv` está comentado en el `__main__`; si se necesitan, basta con descomentar esas líneas.

## `P2_exp.py`

Este archivo importa las funciones de `P2.py` y genera las figuras de los mapas de `c` para el análisis de parámetros. Todas las figuras se guardan en la carpeta `Resultados_P2`. Cada figura tiene tres filas: el mapa de `c` en la primera iteración, el mapa de `c` en la última iteración y la imagen resultante con su RMSE, con una columna por cada valor probado. Los títulos indican los parámetros usados.

Se definieron las siguientes funciones:
1. *mapas_cTV*: Recibe la tupla `es_mapas` con los valores de `e` a probar (por defecto `(0.01, 0.2, 1, 3)`). Los demás parámetros (semilla, imagen, `paso_temporal`, `N`, etc.) se fijan dentro de la función. Para cada `e` llama a *difusion_ansitropica* con `cVT`, calcula el RMSE respecto de la imagen original y grafica los mapas con `imshow` (mapa de colores `viridis`). Como el máximo de `c` es `1/e`, se grafica `e · c` (entre 0 y 1), de modo que todos los mapas compartan la misma escala de color. Guarda la figura como `mapas_cTV.png`.

2. *mapas_cTV_lambda*: Análoga a *mapas_cTV*, pero repite la figura para cada paso temporal de la tupla `pasos_temporales` (`0.01, 0.05, 0.1, 1`), fijando `N`. Sirve para estudiar el efecto conjunto de `e` y λ, incluyendo pasos que superan la cota de estabilidad. Guarda una figura por paso, como `mapas_cTV_lambda_<paso>.png`.

3. *mapas_cp1_lambda*: Para `c_p1`, grafica los mapas para distintos valores del umbral de contraste `k` (la tupla `es_mapas`), con un paso temporal y un `N` fijos. Guarda la figura como `mapas_cp1_lambda_<paso>.png`.

4. *mapas_cp2_lambda*: Análoga a *mapas_cp1_lambda*, pero para `c_p2`. Guarda la figura como `mapas_cp2_lambda_<paso>.png`.

### Uso de `P2_exp.py`:
Se puede ejecutar el archivo directamente. Es necesario tener `cameraman.png` en la misma carpeta. Las funciones se llaman al final del archivo, y para elegir cuáles generar basta con comentar o descomentar las llamadas. En el estado actual del archivo, solo *mapas_cTV* está activa, y como se llama a nivel de módulo, también se ejecuta si el archivo se importa. Los parámetros de cada experimento (semilla, `paso_temporal`, `N`, umbrales y peso) se modifican dentro de cada función.

## `P2_calcular_iteraciones.py`

Importa las funciones de `P2.py` y guarda el RMSE de cada combinación de parámetros en un `.csv`. Se definieron las siguientes variables y funciones:
- Tuplas de valores a probar: `seeds`, `fracciones` (fracción de la cota de estabilidad, de modo que `paso_temporal = fraccion · 1/(4·c_max)`), `es`, `umbrales_de_contrastes`, `umbrales_laplaciano`, `pesos_laplaciano`, `Ns` y `funciones_c`. Se incluye la fracción `1.4` a propósito, para observar qué pasa al superar la cota de estabilidad (por lo que es esperable que aparezcan advertencias).

1. *es_repetida*: Recibe la función c y los parámetros de la combinación actual. Como cada función ignora algunos parámetros, retorna `True` si la corrida sería idéntica a otra ya hecha, para que solo se corra el primer valor de los parámetros que la función no usa: `cVT` solo corre con el primer `k`, `rho` y peso; `c_p1` con el primer `e` y `rho`; y `c_p2` con el primer `e`.

Luego, para cada semilla se crea la carpeta `Resultados_P2/seed_<semilla>/<imagen>`, se carga la imagen, se le agrega el ruido y se calcula el RMSE de la imagen ruidosa. Después se recorren con bucles `for` anidados todas las combinaciones, saltando las repetidas. Para cada una se calcula `c_max` (`1/e` en `cVT` y `1` en las otras dos), el paso temporal, se llama a *difusion_ansitropica*, se calcula el RMSE de la imagen difusa y se guardan dos filas en `resultados.csv` con *guardar_fila* (la difusa y la ruidosa). Se imprime en consola el avance (`hechas/total`).

### Uso de `P2_calcular_iteraciones.py`:
Se puede ejecutar el archivo directamente. Se modifican las tuplas definidas al inicio del archivo. Con los valores actuales hay 980 corridas por semilla (2940 en total para las 3 semillas), por lo que puede demorar. Al igual que en `P1_exp.py`, las filas se agregan al `.csv` sin borrarlo, por lo que si se vuelve a ejecutar con la misma semilla hay que eliminar antes el `resultados.csv` correspondiente.

Los `.csv` generados son los que se usan para graficar RMSE en función de `N`, λ o los demás parámetros, y para elegir los valores de las figuras de `P2_exp.py`.