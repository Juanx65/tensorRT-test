import os
import time

# definir la cantidad de veces que se ejecutarán los programas
num_ejecuciones = 100

# programa 1
tiempo_total_programa1 = 0
for i in range(num_ejecuciones):
    tiempo_inicio = time.time()
    os.system('python eval.py')
    tiempo_fin = time.time()
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    tiempo_total_programa1 += tiempo_ejecucion

tiempo_promedio_programa1 = tiempo_total_programa1 / num_ejecuciones
print(f"Tiempo promedio de eval: {tiempo_promedio_programa1:.2f} segundos")

# programa 2
tiempo_total_programa2 = 0
for i in range(num_ejecuciones):
    tiempo_inicio = time.time()
    os.system("python evalRT.py --engine='weights/mnist_fp16_seg.engine'")
    tiempo_fin = time.time()
    tiempo_ejecucion = tiempo_fin - tiempo_inicio
    tiempo_total_programa2 += tiempo_ejecucion

tiempo_promedio_programa2 = tiempo_total_programa2 / num_ejecuciones
print(f"Tiempo promedio del evalRT: {tiempo_promedio_programa2:.2f} segundos")