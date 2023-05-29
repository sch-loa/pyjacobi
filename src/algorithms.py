import numpy as np
import pandas as pd
import re

DESCOMPOSICION_CARTEL = """\033[F
 __________________________
|                          |  
| DESCOMPOSICION DE JACOBI |
|__________________________|
                        """
RESOLUCION_CARTEL = """\033[F
 _________________________________
|                                 |  
| RESOLUCION DE LA DESCOMPOSICION |
|_________________________________|
                        """

DATAFRAME_VECTOR = pd.DataFrame()
# Funcion principal del metodo de Jacobi,
def jacobi(A, B, k):
    global DESCOMPOSICION_CARTEL
    global RESOLUCION_CARTEL

    # Calcula las matrices necesarias para las operaciones
    L, D, U = calculador_LDU(A)
    H, v = calculador_Hv(L, D, U, B)

    x = np.zeros(B.shape[0]) # Vector de x inicial

    dict_LDU = {' MATRIZ L:': L,' MATRIZ D:': D,' MATRIZ U:': U}
    dict_Hv = {' MATRIZ H:': H, ' MATRIZ V:': v}

    # Imprime las matrices calculadas
    imprimir_matrices_formateadas(DESCOMPOSICION_CARTEL, dict_LDU)
    imprimir_matrices_formateadas(RESOLUCION_CARTEL, dict_Hv)
    print(f' NORMA DE H: {round(np.linalg.norm(H), 2)} \n')

    return iterador_jacobi(A, H, v, x, k)

# Calcula el valor del vector x para cada iteración,
# itera k veces o hasta alcanzar un valor máximo, (si este
# se repite más de una vez, alcanzó el resultado final).
def iterador_jacobi(A, H, v, x, k):
    global DATAFRAME_VECTOR

    x0 = x
    for i in range(1, k+1):
        x1 = np.sum((H * x0) + v, axis = 1) # Nueva aproximacion de x
        Ax1 =  np.sum(A * x0, axis = 1) # Se evalua la aproximacion en la matriz
        
        if(not(np.array_equal(np.round(x0, decimals = 6),np.round(x1, decimals = 6)))):
            DATAFRAME_VECTOR = actualizar_dataframe(DATAFRAME_VECTOR, i, x1, Ax1)

        x0 = x1

    return DATAFRAME_VECTOR, x1, Ax1

# Calcula las siguientes matrices (en el mismo orden):
# L (estrictamente triangular inferior)
# D (diagonal) 
# U (estrictamente triangular superior)
def calculador_LDU(A):
    return (np.tril(A, k = -1), np.diag(np.diag(A)), np.triu(A, k = 1))

# Calcula las matrices H y v producto de operaciones entre las matrices L, D, U, B
def calculador_Hv(L, D, U, B):
    D_inverse = np.linalg.inv(D)
    H = -np.dot(D_inverse, (L + U))
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
        print(f'  {np.round(row, decimals = 2)}')
    print()

# Actualiza los valores del DataFrame con los resultados de la ecuación matricial
# y su evaluación en la matriz inicial.
def actualizar_dataframe(df, i, x, Ax):
    dic = pd.DataFrame({
        'Iteración': i,
        'Aproximación de x': str(np.round(x, decimals = 6)),
        'Evaluación de x en A': str(np.round(Ax, decimals = 6))
    } , index = range(1))
   
    return pd.concat([df, dic], axis = 0, ignore_index = True)

#Convierte la una colección de números en una lista
# usando expresiones regulares
def a_lista(array_strings):
    occurs = list()
    for fila in array_strings: 
        occur = re.findall(r"\d+(?:\.\d+)?", fila)
        occur = np.array([float(oc) for oc in occur])
        occurs.append(occur)
    return occurs
