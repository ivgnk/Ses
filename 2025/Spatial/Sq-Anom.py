"""
d1 - Зависимость числа аномалий от площади работ
"""
import numpy as np
from numpy.linalg import lstsq
import pandas as pd
import matplotlib.pyplot as plt
from rich.jupyter import display
from sympy.printing.pretty.pretty_symbology import line_width
from sympy.strategies import typed

from pcorr import *
#-----------
n_met=['Сейсморазведка', 'Гравиразведка']

nmn = ['Структуры и ПУ', 'Гравитационные аномалии']
nms = ['Площади структур и ПУ', 'Площади гравитационных аномалий']
#----------- column names
cnm=['Название' , 'Площадь, кв. км', 'Число структур и приподнятых участков', 'Число гравитационных аномалий',
     'Площадь структур и ПУ, кв. км', 'Площадь гравитационных аномалий, кв. км']
data=[
['Юго-Камская',   1481, 52, 82, 246.63, 275.3],
['Керчевская',    2848, 29, 65, 282.00, 449.5],
['Пономаревская', 4277, 55, 75, 485.07, 709.2],
['Григорьевская', 1549, 60, 60, 139.14, 439.3],
['Суксунская',    3776, 73, 71, 386.77, 819.51],
['Березовская',   2014, 65, 65, 156.14, 271.55],
['Унинская',      3212, 20, 25, 600.4, 748.29],
['Лимоновско-Вожгальская', 2801, 20, 35, 455.5, 547.34]]
#----------- Для корреляций по числу аномалий
d_nm_s_c=['Название' , 'Площадь, кв. км', 'Число структур и приподнятых участков']
d_nm_s_r=['Юго-Камская', 'Григорьевская', 'Суксунская', 'Березовская']
d_num_s_r=[i for i in range(len(data)) if data[i][0] in d_nm_s_r]
d_nm_g_c=['Название' , 'Площадь, кв. км', 'Число гравитационных аномалий']
d_nm_g_r=['Керчевская', 'Пономаревская', 'Григорьевская', 'Суксунская', 'Березовская']
d_num_g_r=[i for i in range(len(data)) if data[i][0] in d_nm_g_r]
#----------- Для корреляций по площади
d_sq_s_c=['Название' , 'Площадь, кв. км', 'Площадь структур и ПУ, кв. км']
d_sq_g_c=['Название' , 'Площадь, кв. км', 'Площадь гравитационных аномалий, кв. км']

def d1(view_gr=False):
    print('\n',inspect.currentframe().f_code.co_name, """ Зависимость числа аномалий от площади работ""")
    df = pd.DataFrame(data)
    df.columns=cnm
    n=8
    plt.figure(figsize=(n+2,n+3))
    x=df[cnm[1]]
    ys=df[cnm[2]]
    yg=df[cnm[3]]
    if view_gr:
        # Сейсморазведка - red
        plt.plot(x, ys, 'ro', label=nmn[0])
        # Гравиразведка - blue
        plt.plot(x, yg, 'bx', label=nmn[1])
        nms=df[cnm[0]]
        for i, label in enumerate(nms):
            dx = 0;  dy = 1
            if label=='Унинская':
                dy=-2
            elif label=='Лимоновско-Вожгальская':
                dx=-5
            plt.annotate(label, (x[i]+dx, ys[i]+dy), color='red')

        # Гравиразведка
        for i, label in enumerate(nms):
            plt.annotate(label, (x[i], yg[i]-2), color='blue')

        plt.xlim((1410,4800))
        s=12; c='k'
        plt.xlabel(cnm[1], color=c, fontsize=s, fontweight='bold')
        plt.ylabel(cnm[2]+',\n'+cnm[3], color=c, fontsize=s, fontweight='bold')
        plt.title('Зависимость числа аномалий от площади работ')
        plt.legend(loc='upper center'); plt.grid()
    #-------- geeksforgeeks.org/exploring-correlation-in-python/
    """ Зависимость числа аномалий от площади работ"""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    """ Сейсморазведка """
    print(n_met[0])
    df_n_s=df[d_nm_s_c].loc[d_num_s_r, :]
    print(df_n_s.head(20))

    x=np.array(df_n_s['Площадь, кв. км'])
    y=np.array(df_n_s['Число структур и приподнятых участков'])
    print(f'{n_met[0]}  {nmn[0]}: коэффициент корреляции Пирсона XY= {Pearson_correlation(x,y)}')
    coefficients = np.polyfit(x, y, 1)
    a = round(float(coefficients[0]), 4)
    b = round(float(coefficients[1]), 4)
    # print(f"Line Solution is y = {a}x + {b}")
    xl = np.linspace(np.min(x), np.max(x))
    yl = a*xl + b
    if view_gr:
        print(x,y)
        plt.scatter(x, y, s=100, edgecolors='r', facecolors='none')
        plt.plot(xl,yl,'--', color='r', linewidth=0.5)

    """ Гравиразведка """
    print('\n',n_met[1])
    df_n_g = df[d_nm_g_c].loc[d_num_g_r, :]
    print(df_n_g.head(20))

    x=np.array(df_n_g['Площадь, кв. км'])
    y=np.array(df_n_g['Число гравитационных аномалий'])
    print(f'{n_met[1]}  {nmn[1]}: коэффициент корреляции Пирсона XY= {Pearson_correlation(x,y)}')

    coefficients = np.polyfit(x, y, 1)
    a = round(float(coefficients[0]), 4)
    b = round(float(coefficients[1]), 4)
    # print(f"Line Solution is y = {a}x + {b}")
    xl = np.linspace(np.min(x), np.max(x))
    yl = a*xl + b
    if view_gr:
        print(x,y)
        plt.scatter(x, y, s=250, edgecolors='b', facecolors='none')
        plt.plot(xl,yl,'--', color='b', linewidth=0.5)
        plt.show()



def coeff_corr(df:pd.DataFrame):
    pass


def d2(view_gr=False):
    print('\n',inspect.currentframe().f_code.co_name,""" Зависимость площади аномалий от площади работ """)
    df = pd.DataFrame(data)
    df.columns=cnm
    n=8
    plt.figure(figsize=(n+2,n+3))
    x=df[cnm[1]]
    ys=df[cnm[4]]
    yg=df[cnm[5]]
    if view_gr:
        # Сейсморазведка - red
        plt.plot(x, ys, 'ro', label=nmn[0])
        # Гравиразведка - blue
        plt.plot(x, yg, 'bx', label=nmn[1])
        nms=df[cnm[0]]
        for i, label in enumerate(nms):
            dx = 0;  dy = 10
            if label=='Унинская':
                pass
                # dy=-2
            elif label=='Лимоновско-Вожгальская':
                pass
                # dx=-5
            plt.annotate(label, (x[i]+dx, ys[i]+dy), color='red')

        # Гравиразведка
        for i, label in enumerate(nms):
            plt.annotate(label, (x[i]+25, yg[i]), color='blue')

        plt.xlim((1410,4800))
        s=12; c='k'
        plt.xlabel(cnm[1], color=c, fontsize=s, fontweight='bold')
        plt.ylabel(cnm[4]+',\n'+cnm[5], color=c, fontsize=s, fontweight='bold')
        plt.title('Зависимость площади аномалий от площади работ')
        plt.legend(loc='upper center'); plt.grid()
    # -------- geeksforgeeks.org/exploring-correlation-in-python/
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    """ Сейсморазведка """
    print(n_met[0])
    x=np.array(df['Площадь, кв. км'])
    y=np.array(df['Площадь структур и ПУ, кв. км'])
    print(f'{n_met[0]}  {nmn[0]}: коэффициент корреляции Пирсона XY= {Pearson_correlation(x,y)}')
    coefficients = np.polyfit(x, y, 1)
    a = round(float(coefficients[0]), 4)
    b = round(float(coefficients[1]), 4)

    xl = np.linspace(np.min(x), np.max(x))
    yl = a*xl + b
    if view_gr:
        print(x,y)
        # plt.scatter(x, y, s=100, edgecolors='r', facecolors='none')
        plt.plot(xl,yl,'--', color='r', linewidth=0.5)

    """ Гравиразведка """
    print('\n',n_met[1])
    x=np.array(df['Площадь, кв. км'])
    y=np.array(df['Площадь гравитационных аномалий, кв. км'])
    print(f'{n_met[1]}  {nmn[1]}: коэффициент корреляции Пирсона XY= {Pearson_correlation(x,y)}')

    coefficients = np.polyfit(x, y, 1)
    a = round(float(coefficients[0]), 4)
    b = round(float(coefficients[1]), 4)
    # print(f"Line Solution is y = {a}x + {b}")
    xl = np.linspace(np.min(x), np.max(x))
    yl = a*xl + b
    if view_gr:
        print(x,y)
        # plt.scatter(x, y, s=250, edgecolors='b', facecolors='none')
        plt.plot(xl,yl,'--', color='b', linewidth=0.5)
        plt.show()


if __name__=="__main__":
    # d1(True)
    d2(True)