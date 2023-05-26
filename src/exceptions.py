class ColumnSizeDifference(Exception):
    def __init__(self):
        super.__init__("El número de columnas de los elementos matriciales no coincide.")

class NegativeNumber(Exception):
    def __init__(self):
        super().__init__(f"El campo numérico debe ser positivo.")

def is_column_size_different(A_size, B_size):
    if(A_size != B_size):
        raise ColumnSizeDifference

def is_zero_or_natural(value):
    if(value < 0):
        raise NegativeNumber()
