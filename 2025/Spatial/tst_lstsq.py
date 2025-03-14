"""
Тестирование - numpy.linalg  lstsq
"""
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

def test1():
    """
    stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
    """
    points = [(1,5),(3,4)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    print(f'{A=}')
    print(f'{y_coords=}')
    m, c = lstsq(A, y_coords, rcond=None)[0]
    print("Line Solution is y = {m}x + {c}".format(m=m,c=c))

    print()
    print(f'{x_coords=}')
    print(f'{y_coords=}')

    coefficients = np.polyfit(x_coords, y_coords, 1)
    # Print the findings
    a=round(coefficients[0],29)
    print('a =', a)
    b=round(coefficients[1],29)
    print('b =', b)
    print(f"Line Solution is y = {a}x + {b}")
    plt.plot(x_coords, y_coords)
    plt.plot(x_coords, y_coords,'ro')
    plt.grid(); plt.show()

def the_test2():
    x=[1481, 1549, 3776, 2014]
    y=[52, 60, 73, 65]
    test2(x, y)
    x1=[1481, 1549, 2014, 3776]
    y1=[52, 60, 65, 73]
    test2(x1, y1)

def test2(x, y):
    coefficients = np.polyfit(x, y, 1)
    print("Linear polynomial coefficients:", coefficients)
    # Print the findings
    a=float(coefficients[0])
    print('a =', a)
    b=float(coefficients[1])
    print('b =', b)
    print(f"Line Solution is y = {a}x + {b}")
    plt.plot(x, y,'ro')
    xl = np.linspace(np.min(x), np.max(x))
    yl = a*xl + b

    plt.plot(xl, yl, '--', color='r')
    plt.grid(); plt.show()

if __name__=='__main__':
    the_test2()