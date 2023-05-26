import numpy as np
import pandas as pd

from exceptions import is_column_size_different, is_zero_or_natural
from algorithms import jacobi, imprimir_matriz

METODO_CARTEL = """
 _____________________
|                     |
| METODO DE JACOBI    |
|_____________________|
                """

A_matrix = np.array([[2,1],[5,7]])
B_vector = np.array([11,13])

A_SHAPE = A_matrix.shape[0]
B_SHAPE = B_vector.shape[0]
is_column_size_different(A_SHAPE, B_SHAPE)

print(METODO_CARTEL)

print(' MATRIZ A:')
imprimir_matriz(A_matrix)

print(' MATRIZ B:')
print('  ' + str(B_vector))
print()

print(' NUMERO DE ITERACIONES:')
k_iters = int(input(' |_k: ')) # Número de iteraciones
is_zero_or_natural(k_iters)
print()

datos_vector, x, Ax = jacobi(A_matrix, B_vector, A_SHAPE, k_iters)
print(datos_vector.to_string(index = False))

print(f"\nSolución final aproximada: {str(np.round(x, decimals = 4))}")
print("Solución mediante un método exacto: " + str(np.round(np.linalg.solve(A_matrix, B_vector), decimals = 4)))