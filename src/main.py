import numpy as np

from exceptions import is_column_size_different, is_zero_or_natural
from algorithms import jacobi, print_matrix

A_matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
B_vector = np.array([11,22,33])

A_shape = A_matrix.shape[0]
B_shape = B_vector.shape[0]
is_column_size_different(A_shape, B_shape)

print("MATRIZ A")
print_matrix(A_matrix)
print()

print("VECTOR B")
print(B_vector)
print()

print("NUMERO DE ITERACIONES")
k_iters = int(input("|_k: ")) # Número de iteraciones
is_zero_or_natural(k_iters)

jacobi(A_matrix, B_vector, A_shape, k_iters)