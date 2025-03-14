import sys
import inspect
import numpy as np
def Pearson_correlation(X,Y):
    """
    Коэффициент корреляцуии Пирсона
    www.geeksforgeeks.org/exploring-correlation-in-python/
    """
    if len(X)==len(Y):
        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
        Sum_x_squared = sum((X-X.mean())**2)
        Sum_y_squared = sum((Y-Y.mean())**2)
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    else:
        print(f"Function = {inspect.currentframe().f_code.co_name}")
        print('Массивы X и Y разной дляины')
        sys.exit()
    return corr