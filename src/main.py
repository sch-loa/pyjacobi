import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from exceptions import is_column_size_different, is_square, is_zero_or_natural, is_jacobi_operable
from algorithms import jacobi, imprimir_matriz, a_lista

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
|  |_ Ax = B, siendo:      ____________        ___        ____   |
|                     A = |  3  -1  -1 |  B = | 1 |  x = | x0 |  |
|                         | -1   3   1 |      | 3 |      | x1 |  |
|                         |  2   1   4 |      | 7 |      | x2 |  |
|                         |____________|      |___|      |____|  |
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
|  anterior, indica que como mínimo la solución se ha            | 
|  estabilizado y no se producen cambios significativos en las   |
|  siguientes iteraciones. Por lo tanto se toma como solución    |
|  final. Dado que se eligió representar los datos hasta 6       |
|  decimales, se toma esta cifra para las comparaciones de los   |
|  valores hallados, ya que seguir mostrando el resultado de     |
|  las iteraciones no aporta información real. Algo a tener en   |
|  cuenta es que si se pide un número k de iteraciones y la      |
|  aproximación final es alcanzada con n < k, el programa itera  |
|  hasta esa cifra.                                              |
|________________________________________________________________|
                """

A_matrix = np.array([[3,-1,-1],[-1,3,1],[2,1,4]])
is_jacobi_operable(A_matrix) # Verifico que la matriz sea estrictamente dominante
B_vector = np.array([1,3,7])

is_square(A_matrix) # Verifico que la matriz sea cuadrada
# Verifico que los vectores sean de tamaños equivalentes
is_column_size_different(A_matrix, B_vector)

print(METODO_CARTEL)
print(' MATRIZ A:')
imprimir_matriz(A_matrix)

print(' VECTOR B:')
print('  ' + str(B_vector))
print()

print(' NUMERO DE ITERACIONES:')
k_iters = int(input(' |_k: '))
is_zero_or_natural(k_iters)
print()

# Extraigo DataFrame con datos de las iteraciones, vector
# aproximado de x y evaluación del vector en A
datos_vector, x, Ax = jacobi(A_matrix, B_vector, k_iters)
print(datos_vector.to_string(index = False))

# Imprimo resultados finales
## RESULTADOS APROXIMADOS
print(f"\nSolución final aproximada: {np.round(x, decimals = 4)}")
print(f" |_Evaluación en A: {np.round(Ax, decimals = 4)}\n")
## RESULTADOS FINALES
x_exactos = np.linalg.solve(A_matrix, B_vector)
print(f"Solución mediante un método exacto: {np.round(x_exactos, decimals = 4)}")
print(f" |_Evaluación en A: {np.round(np.sum(A_matrix * x_exactos, axis = 1), decimals = 4)}")

x_vals = a_lista(datos_vector['Aproximación de x'].values)
y_vals = a_lista(datos_vector['Evaluación de x en A'].values)

# GRÁFICOS DE CONVERGGENCIA DEL METODO
TITULOS = ['X0', 'X1', 'X2']
COLORES = ['r', 'c', 'y']
for i in range(B_vector.shape[0]):
    # Valores aproximados de x e y
    x_subvals = [j[i] for j in x_vals]
    y_subvals = [j[i] for j in y_vals]

    # Valores reales x e y
    x_sub = x[i]
    y_sub = Ax[i]

    fig, ax = plt.subplots()

    # Ajusto la perspectiva del gráfico para hacer foco en el valor real
    plt.xlim(x_sub-5, x_sub+5)
    plt.ylim(y_sub-5, y_sub+5)

    ax.scatter(x_subvals, y_subvals, color = COLORES[i])
    ax.scatter([x_sub], [y_sub], color = 'k')

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray') 
    plt.title(f'Convergencia del Método en {TITULOS[i]}')

    plt.show()
