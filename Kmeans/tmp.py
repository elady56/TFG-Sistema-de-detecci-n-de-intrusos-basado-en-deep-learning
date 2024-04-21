import time
import random
import sys

for i in range(20):
    # Obtener el tiempo actual en segundos desde la época
    time1 = int(time.time())

    # Inicializar el generador de números aleatorios con la semilla obtenida del tiempo actual
    random.seed(time1)

    # Generar un número aleatorio
    random_number = random.randint(0, sys.maxsize)

    # Imprimir el número aleatorio y el tiempo actual
    print(random_number)
    print(time1)

    # Esperar 2 segundos
    time.sleep(2)
