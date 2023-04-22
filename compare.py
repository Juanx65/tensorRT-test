import os
import time

# definir la cantidad de veces que se ejecutarán los programas
num_ejecuciones = 1000

# programa 0
tiempo_total_programa0 = 0
tiempo_minimo_programa0 = float("inf")
tiempo_maximo_programa0 = float("-inf")
for i in range(num_ejecuciones):
    tiempo_inicio = time.time()
    os.system("python eval.py")
    tiempo_fin = time.time()
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    tiempo_total_programa0 += tiempo_ejecucion

    # Actualizar el tiempo mínimo y máximo
    tiempo_minimo_programa0 = min(tiempo_minimo_programa0, tiempo_ejecucion)
    tiempo_maximo_programa0 = max(tiempo_maximo_programa0, tiempo_ejecucion)


tiempo_promedio_programa0 = tiempo_total_programa0 / num_ejecuciones
print(f"Tiempo promedio de eval vanilla: {tiempo_promedio_programa0:.2f} segundos")
print(f"Tiempo mínimo de eval vanilla: {tiempo_minimo_programa0:.2f} segundos")
print(f"Tiempo máximo de eval vanilla: {tiempo_maximo_programa0:.2f} segundos")

# programa 1
tiempo_total_programa1 = 0
tiempo_minimo_programa1 = float("inf")
tiempo_maximo_programa1 = float("-inf")
for i in range(num_ejecuciones):
    tiempo_inicio = time.time()
    os.system("python evalRT.py --engine='weights/mnist_fp32.engine'")
    tiempo_fin = time.time()
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    tiempo_total_programa1 += tiempo_ejecucion

    # Actualizar el tiempo mínimo y máximo
    tiempo_minimo_programa1 = min(tiempo_minimo_programa1, tiempo_ejecucion)
    tiempo_maximo_programa1 = max(tiempo_maximo_programa1, tiempo_ejecucion)


tiempo_promedio_programa1 = tiempo_total_programa1 / num_ejecuciones
print(f"Tiempo promedio de evalRT fp32: {tiempo_promedio_programa1:.2f} segundos")
print(f"Tiempo mínimo de evalRT fp32: {tiempo_minimo_programa1:.2f} segundos")
print(f"Tiempo máximo de evalRT fp32: {tiempo_maximo_programa1:.2f} segundos")

# programa 2
tiempo_total_programa2 = 0
tiempo_minimo_programa2 = float("inf")
tiempo_maximo_programa2 = float("-inf")
for i in range(num_ejecuciones):
    tiempo_inicio = time.time()
    os.system("python evalRT.py --engine='weights/mnist_fp16_seg.engine'")
    tiempo_fin = time.time()
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    tiempo_total_programa2 += tiempo_ejecucion

    # Actualizar el tiempo mínimo y máximo
    tiempo_minimo_programa2 = min(tiempo_minimo_programa2, tiempo_ejecucion)
    tiempo_maximo_programa2 = max(tiempo_maximo_programa2, tiempo_ejecucion)

tiempo_promedio_programa2 = tiempo_total_programa2 / num_ejecuciones
print(f"Tiempo promedio del evalRT fp16: {tiempo_promedio_programa2:.2f} segundos")
print(f"Tiempo mínimo de evalRT fp16: {tiempo_minimo_programa2:.2f} segundos")
print(f"Tiempo máximo de evalRT fp16: {tiempo_maximo_programa2:.2f} segundos")