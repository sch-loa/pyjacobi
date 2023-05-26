import numpy as np

def jacobi(A, B, n, k):
    L = np.tril(A, k = -1) # Se transforma en estrictamente triangular inferior
    D = np.diag(np.diag(A)) # Se transforma en diagonal
    U = np.triu(A, k = 1) # Se transforma en estrictamente triangular superior
    print_matrix(L)
    print()
    print_matrix(U)

def jacobi_it(L, D, U, k):
    H = 0

def print_matrix(matrix):
    for row in matrix:
        print(row)

