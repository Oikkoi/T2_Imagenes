# P1 (`P1.py`):
desviación estándar = 0 es la identidad
al final, en la sección de `if __name__ == "__main__":` (línea 78) nos encontramos los dos parámetros modificables: semilla y desviacion_estandar.

Por simplicidad en el diseño, se espera que se borre la carpeta /Resultados_P1 completa entre ejecuciones de P1_exp y P1_gráficos. El flujo esperado es: Ejecutar P1_exp.py, luego P1_gráficos.py. 

Resultados/P1 surge al ejecutar el bloque `if __name__ == "__main__"` en P1_exp.py

exp 3: Obviamente no comparten el mismo mínimo, pero a grandes rasgos parece que la intensidad influye en la RMSE (por eso el fondo siempre es el que tiene el mínimo mayor, porque tiene la intensidad más notablemenete baja) y la cantidad de pixeles tiene algo que ver, porque generlamente es la imagen completa la que tiene el segundo mínimo más alto. Si tuviera que generalizar, mientras más cerca del centro esté una intensidad, menos se nota el filtro de ruido y por eso tiene una RMSE más baja (es decir, el cuadrado). Pero mientras más pixeles tiene, hay más espacio para que haya ruido (es decir, el cuadrado completo que suele tener la segunda mayor cantidad de ruido)

librerías:
- cv2
- matplotlib
- skimage
- numpy