import numpy as np

def jacobi(A, B, k):
    L, D, U = L_D_U_calculator(A)
    H, v = H_v_calculator(L, D, U, B)



def L_D_U_calculator(A):
    return (np.tril(A, k = -1), np.diag(np.diag(A)), np.triu(A, k = 1))

def H_v_calculator(L, D, U, B):
    minus_D_inverse = -np.linalg.inv(D)
    return (minus_D_inverse + (L + U), minus_D_inverse * B)

def print_matrix(matrix):
    for row in matrix:
        print(row)

