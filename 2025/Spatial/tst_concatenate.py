import numpy as np
"""
объединить три столбца в матрицу с помощью NumPy
функцию np.concatenate()
https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
"""
def tst_col_unite():
    mat1 = np.array([413.3367, 413.3787, 413.4127, 413.4552, 413.5062])
    mat2 = np.array([389.003,  388.622,  388.3935, 388.207,  387.9865])
    currl=len(mat1)
    lblst=[12]*currl
    res =  np.column_stack((mat1, mat2,lblst))
    print(res)
    b=np.zeros((1,3))
    print(b)
    res2=np.vstack((b,res))
    print(res2)
    res3=res2[1:]
    print(res3)

if __name__=="__main__":
    tst_col_unite()