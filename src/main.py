import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from exceptions import is_column_size_different, is_square, is_zero_or_natural, is_jacobi_operable
from algorithms import jacobi, imprimir_matriz, a_lista

METODO_CARTEL = """
 ____________________________________________________________________________
|                                                                            |
|                              METODO DE JACOBI                              |
|____________________________________________________________________________|
|                                                                            |
|  INTEGRANTES:                                                              |
|  |_ Loana Abril Schleich Garcia.                                           |
|                                                                            |
|  SISTEMA DE ECUACIONES A EVALUAR:                                          |
|  |_ Ax = B, siendo:                                                        |
|____________________________________________________________________________|
|                                                                            |
|                        FUNCIONAMIENTO DEL ALGORITMO                        |
|____________________________________________________________________________|
|                                                                            |
|  Para aproximar el valor de x se transforma la matriz A en la suma de      | 
|  tres partes:                                                              |
|                                                                            |
|  L -> Estrictamente triangular inferior                                    |
|  D -> Diagonal                                                             |
|  U -> Estrictamente triangular superior                                    |
|                                                                            |
|  Producto de las cuales por medio de otras operaciones se obtienen dos     |
|  matrices H y v, tal que:                                                  |
|                                                                            |
|  -> H = -D^-1 * (L + U)                                                    |
|  -> v = D^-1 * B                                                           |
|                           x_(k+1) = (H * x_k) + v                          |
|                                                                            |
|  Siendo k el número de iteraciones a realizar del método, y x_k una        |
|  primera aproximación arbitraria de la incógnita matricial. La operación   |
|  se realiza k veces, o hasta haberse alcanzado una aproximación final.     |
|  Si se cumple que el resultado de una aproximación es exactamente igual a  |
|  la anterior, indica que como mínimo la solución se ha estabilizado y no   |
|  se producen cambios significativos en las siguientes iteraciones. Por lo  |
|  tanto se toma como solución final. Dado que se eligió representar los     |
|  datos hasta 6 decimales, se toma esta cifra para las comparaciones entre  |
|  los valores hallados, ya que seguir mostrando el resultado de las         |
|  iteraciones no aporta información real.                                   |
|____________________________________________________________________________|
                """

A_matrix = np.array([[3,-1,-1],[-1,3,1],[2,1,4]])
is_jacobi_operable(A_matrix) # Verifico que la matriz sea estrictamente dominante
print('La matriz es estrictamente dominante, es posible operar con el algoritmo de Jacobi.')
B_vector = np.array([1,3,7])

is_square(A_matrix) # Verifico que la matriz sea cuadrada
# Verifico que los vectores sean de tamaños equivalentes
is_column_size_different(A_matrix, B_vector)

print(METODO_CARTEL)
print(' MATRIZ A:')
imprimir_matriz(A_matrix)

print(' VECTOR B:')
print(f'  {B_vector}')
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
print(f"\n\n SOLUCION FINAL APROXIMADA: {np.round(x, decimals = 4)}")
print(f"  |_Evaluación en A: {np.round(Ax, decimals = 4)}\n")
## RESULTADOS FINALES
x_exactos = np.linalg.solve(A_matrix, B_vector)
y_exactos = np.sum(A_matrix * x_exactos, axis = 1)
print(f" SOLUCION MEDIANTE UN METODO EXACTO: {np.round(x_exactos, decimals = 4)}")
print(f"  |_Evaluación en A: {np.round(y_exactos, decimals = 4)}\n")

x_vals = a_lista(datos_vector['Aproximación de x'].values)
y_vals = a_lista(datos_vector['Evaluación de x en A'].values)

# GRÁFICOS DE CONVERGENCIA DEL METODO
COLORES = ['r', 'c', 'y']
for i in range(B_vector.shape[0]):
    # Valores aproximados de x e y
    x_subvals = [j[i] for j in x_vals]
    y_subvals = [j[i] for j in y_vals]

    # Valores reales x e y
    x_sub = x_exactos[i]
    y_sub = y_exactos[i]

    fig, ax = plt.subplots()

    # Ajusto la perspectiva del gráfico para hacer foco en el valor real
    plt.xlim(x_sub-4, x_sub+4)
    plt.ylim(y_sub-4, y_sub+4)

    ax.scatter(x_subvals, y_subvals, color = COLORES[i])
    ax.scatter([x_sub], [y_sub], color = 'k')

    plt.grid(True, linestyle='--', linewidth=0.3, color='gray') 
    plt.title(f'Acercamiento a X{i} en cada Iteración')

    plt.show()
