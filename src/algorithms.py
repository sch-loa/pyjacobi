import numpy as np
import pandas as pd

DESCOMPOSICION_CARTEL = """
 __________________________
|                          |  
| DESCOMPOSICION DE JACOBI |
|__________________________|
                        """
RESOLUCION_CARTEL = """
 _________________________________
|                                 |  
| RESOLUCION DE LA DESCOMPOSICION |
|_________________________________|
                        """

DATAFRAME_VECTOR = pd.DataFrame()

def jacobi(A, B, n, k):
    global DESCOMPOSICION_CARTEL
    global RESOLUCION_CARTEL

    L, D, U = calculador_LDU(A)
    H, v = calculador_Hv(L, D, U, B)
    x = np.ones(n)

    dict_LDU = {' MATRIZ L:': L,' MATRIZ D:': D,' MATRIZ U:': U}
    dict_Hv = {' MATRIZ H:': H, ' MATRIZ v:': v}

    imprimir_matrices_formateadas(DESCOMPOSICION_CARTEL, dict_LDU)
    imprimir_matrices_formateadas(RESOLUCION_CARTEL, dict_Hv)

    return iterador_jacobi(A, H, v, x, k)

def iterador_jacobi(A, H, v, x0, k):
    global DATAFRAME_VECTOR
    for i in range(1, k+1):
        x1 = np.sum((H * x0) + v, axis = 1)
        DATAFRAME_VECTOR = actualizar_dataframe(DATAFRAME_VECTOR, i, x0, x1, np.sum(A, axis = 1))
        x0 = x1
        
    return DATAFRAME_VECTOR

# Calcula las siguientes matrices (en el mismo orden):
# L (estrictamente triangular inferior)
# D (diagonal) 
# U (estrictamente triangular superior)
def calculador_LDU(A):
    return (np.tril(A, k = -1), np.diag(np.diag(A)), np.triu(A, k = 1))

# Calcula las matrices H y v producto de operaciones entre las matrices L, D, U, B
def calculador_Hv(L, D, U, B):
    D_inverse = np.linalg.inv(D)
    H = -1 * D_inverse * (L + U)
    v = D_inverse * B
    return (H, v)

# Imprime un título y matrices con sus respectivos subtitulos
def imprimir_matrices_formateadas(titulo, dict_formateado):
    print(titulo)
    for (nombre, matriz) in dict_formateado.items():
        print(nombre)
        imprimir_matriz(matriz)

# Imprime matriz columna por columna
def imprimir_matriz(matrix):
    for row in matrix:
        print('  ' + str(row))
    print()

# Actualiza los valores del DataFrame con los resultados de la ecuación matricial
# y su evaluación en la matriz inicial.
def actualizar_dataframe(df, i, x, Ax):
    dic = pd.DataFrame({
        'Iteración': i,
        'Matriz aproximada': str(x),
        'Evaluación aproximada': str(Ax)
    } , index = range(1))
   
    return pd.concat([df, dic], axis = 0, ignore_index = True)

