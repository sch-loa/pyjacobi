import numpy as np
import pandas as pd

from exceptions import is_column_size_different, is_square, is_zero_or_natural
from algorithms import jacobi, imprimir_matriz

METODO_CARTEL = """
 ________________________________________________________________
|                                                                |
|                        METODO DE JACOBI                        |
|________________________________________________________________|
|                                                                |
|  INTEGRANTES:                                                  |
|  |_ Loana Abril Schleich Garcia.                               |
|                                                                |
|  SISTEMA DE ECUACIONES A EVALUAR:                              |
|  |_ Ax = B                                                     |
|                                                                |
|________________________________________________________________|
|                                                                |
|                  FUNCIONAMIENTO DEL ALGORITMO                  |
|________________________________________________________________|
|                                                                |
|  Para aproximar el valor del vector x se descompone            |
|  la matriz A en tres partes:                                   |
|                                                                |
|  L -> Estrictamente triangular inferior                        |
|  D -> Diagonal                                                 |
|  U -> Estrictamente triangular superior                        |
|                                                                |
|  Producto de las cuales por medio de otras operaciones se      |
|  obtienen dos matrices H y v, tal que:                         |
|                                                                |
|  x_(k+1) = (H * x_k) + v                                       |
|                                                                |
|  Siendo k el número de iteraciones a realizar del método,      |
|  y x_k una primera aproximación arbitraria de la incognita     |
|  matricial.                                                    |
|  La operación se realiza k veces, o hasta haberse alcanzado    |
|  una aproximación final. En este caso, si se cumple que el     |
|  resultado de una aproximación es exactamente igual a la       |
|  anterior, indica que se ha alcanzado una aproximación total,  |
|  o como mínimo la solución se ha estabilizado y no se          |
|  producen cambios significativos (teniendo en cuenta los       |
|  límites de representación del lenguaje) en las siguientes     |
|  iteraciones. Por lo tanto se toma como solución final.        |
|________________________________________________________________|
                """

A_matrix = np.array([[2,1],[5,7]])
B_vector = np.array([11,13])

is_square(A_matrix) # Verifico que la matriz sea cuadrada
# Verifico que los vectores sean de tamaños equivalentes
is_column_size_different(A_matrix, B_vector)

print(METODO_CARTEL)
print(' MATRIZ A:')
imprimir_matriz(A_matrix)

print(' MATRIZ B:')
print('  ' + str(B_vector))
print()

print(' NUMERO DE ITERACIONES:')
k_iters = int(input(' |_k: '))
is_zero_or_natural(k_iters)
print()

# Extraigo DataFrame con datos de las iteraciones, vector
# aproximado de x y evaluación del vector en A
datos_vector, x, Ax = jacobi(A_matrix, B_vector, B_vector.shape[0], k_iters)
print(datos_vector.to_string(index = False))

# Imprimo resultados finales
## Resultados aproximados
print(f"\nSolución final aproximada: {np.round(x, decimals = 4)}")
print(f" |_Evaluación en A: {np.round(Ax, decimals = 4)}\n")
## Resultados finales
x_exactos = np.linalg.solve(A_matrix, B_vector)
print(f"Solución mediante un método exacto: {np.round(x_exactos, decimals = 4)}")
print(f" |_Evaluación en A: {np.round(np.sum(A_matrix * x_exactos, axis = 1), decimals = 4)}")

# Grafico convergencia del metodo
