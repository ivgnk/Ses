"""
Кластеризация центров грав.аномалий
На основе статьи
Кластеризация пространственных данных – k-means и иерархические алгоритмы
https://cartetika.ru/tpost/uun5jy5tk1-klasterizatsiya-prostranstvennih-dannih
"""
import sys

# from pspat_const import *
from pspat_01 import *
# from pspat_02 import *
# from pspat_clust01 import *
# from pspat_const import *
from pspat_work import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# cluster_for_UKarea:
# mid = import_mid2pandas(UK_mid_fname, name_col)
# xmain, xcoo, ycoo, lbl = expand_mid(mid)
# spat_clust01(xmain)

type_dv=['Словарь, расстояния', 'Словарь, приращение расстояний', 'Матрица, расстояния',  'Модель, матрица, расстояния',]

def main_fun():
    print('\nFunction name = ', inspect.currentframe().f_code.co_name)
    inf_fn_pd = input_inf(f_inf_nm)
    sys.exit()
    # view_inf(inf_fn_pd)
    the_view=2

    for i in range(len(inf_fn_pd)):
        sq_name=list(inf_fn_pd.iloc[[0]]['sq_name'])[0]
        print(f'{sq_name=}')
        mifd_fn=list((inf_fn_pd.iloc[[0]]["f_name"]))[0]
        print(f'{mifd_fn=}')
        gdf=input_mifd(mifd_fn, sq_name, view=True)
        # Определение расстояний между центрами, получение словаря
        dd_ini, dd_cor=fun_CentrXY_01(gdf, view=the_view)
        # Определение расстояний между центрами, получение матрицы
        data1, res = fun_CentrXY_02(gdf)
        cntr_xy_stat(res)
        match the_view:
            case 0: view_dict_length0(dd_ini) # Диаграмма - scatter - диаграмма расстояний
            case 1: view_dict_length2(dd_ini) # График - Вычисление и визуализация нарастающих расстояний или приращений расстояний
            case 2: view_color_grid(data1)  # реальная сетка (треугольник) для сейсмических аномалий Юго-Камской площади
            case 3: # модельная сетка (треугольник) с квадратным расположением аномалий
                    ddat=fun_CentrXY_mod()
                    view_color_grid(ddat)
            case _: print("Undefined"); sys.exit()
        # work_with_mifd(fl_name, sq_name)

if __name__=="__main__":
    main_fun()

# Сделать
#  Поле2D_01_(2024H2).ppt - Scipy.Spatial – пространственные  функции
#  Query
#  query_ball_point
#  query_pairs
#  pdist – попарные расстояния между наблюдениями.1

# Scipy.Cluster – кластеризация
# Кластеризация по CenterX, CenterY потом по Square, Perimeter
# Кластеризация по Square, Perimeter потом по CenterX, CenterY
# https://en.wikipedia.org/wiki/K-means_clustering

# Корреялция по Square, Perimeter


# Двумерное преобразовыавние Фурье


