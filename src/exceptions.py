import numpy as np

class ColumnSizeDifference(Exception):
    def __init__(self):
        super().__init__("El número de columnas de los elementos matriciales debe coincidir")

class NonJacobiOperable(Exception):
    def __init__(self):
        super().__init__("La matriz debe ser estrictamente dominante")

class NonSquareMatrix(Exception):
    def __init__(self):
        super().__init__("La matriz debe ser cuadrada")

class NegativeNumber(Exception):
    def __init__(self):
        super().__init__(f"El campo numérico debe ser positivo")

# Verifica que dos matrices/vectores tengan el mismo
# número de columnas para poder hacer operaciones entre las mismas.
def is_column_size_different(A, B):
    if(A.shape[0] != B.shape[0]):
        raise ColumnSizeDifference

# Verifica que la matriz sea cuadrada
def is_square(A):
    if(A.shape[0] != A.shape[1]):
        raise NonSquareMatrix

# Verifica que un número sea cero o natural
def is_zero_or_natural(value):
    if(value < 0):
        raise NegativeNumber()

# Verifica que el metodo de jacobi sea aplicable en la matriz A,
# para esto verifica que la diagonal de la matriz sea estrictamente
# dominante, es decir que la suma de los valores de la misma (en valor absoluto)
#  sea mayor a la suma de los elementos restantes de la matriz.
def is_jacobi_operable(A):
    diagonal = np.sum(np.abs(np.diag(np.diag(A))))
    matriz_restante = np.sum(np.abs(A)) - diagonal

    if(not np.all(diagonal > matriz_restante)):
        raise NonJacobiOperable