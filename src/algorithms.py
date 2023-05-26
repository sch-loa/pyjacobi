import numpy as np
import pandas as pd

descomposicion_cartel = """
 __________________________
|                          |  
| DESCOMPOSICION DE JACOBI |
|__________________________|
                        """
resolucion_cartel = """
 _________________________________
|                                 |  
| RESOLUCION DE LA DESCOMPOSICION |
|_________________________________|
                        """

def jacobi(A, B, n, k):
    global descomposicion_cartel
    global resolucion_cartel

    L, D, U = calculador_LDU(A)
    H, v = calculador_Hv(L, D, U, B)
    x = np.zeros(n)

    dict_LDU = {' MATRIZ L:': L,' MATRIZ D:': D,' MATRIZ U:': U}
    dict_Hv = {' MATRIZ H:': H, ' VECTOR v:': v}

    imprimir_matrices_formateadas(descomposicion_cartel, dict_LDU)
    imprimir_matrices_formateadas(resolucion_cartel, dict_Hv)

    iterador_jacobi(H, v, x, k)

def iterador_jacobi(H, v, x, k):
    for i in range(k):
        pass

def calculador_LDU(A):
    return (np.tril(A, k = -1), np.diag(np.diag(A)), np.triu(A, k = 1))

def calculador_Hv(L, D, U, B):
    minus_D_inverse = -np.linalg.inv(D)
    H = np.round(minus_D_inverse + (L + U), decimals=2)
    v = np.round(minus_D_inverse * B, decimals=2)
    return (H, v)

def imprimir_matrices_formateadas(titulo, dict_formateado):
    print(titulo)
    for (nombre, matriz) in dict_formateado.items():
        print(nombre)
        imprimir_matriz(matriz)

def imprimir_matriz(matrix):
    for row in matrix:
        print('  ' + str(row))
    print()

