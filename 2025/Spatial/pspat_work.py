"""
pspat_work
Обработка введенных данных

en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset
en.wikipedia.org/wiki/List_of_map_projections
en.wikipedia.org/wiki/SK-42_reference_system
"""
from multiprocessing.sharedctypes import SynchronizedArray

import numpy as np
from matplotlib.pyplot import figure
from scipy.constants import value
from statsmodels.sandbox.regression.try_treewalker import data2
from sympy.abc import alpha
from sympy.physics.units import farad

from pspat_const import *
import geopandas
import sys
import inspect
import matplotlib.pyplot as plt
from math import hypot
from pprint import pp
from math import pi
import statistics
from pspat_anom_dist import *
# from statistics import *

# Определение расстояний между центрами, получение матрицы
def fun_CentrXY_02(gdf:geopandas.geodataframe.GeoDataFrame, view=False):
    """
    Определение расстояний между центрами. Попарные расстояния, матрицы
    """
    print('Function name = ', inspect.currentframe().f_code.co_name)
    ll=len(gdf)
    data=np.zeros((ll,ll))
    res=[]
    for i in range(ll):
        for j in range(ll):
            if i==j or i<j:
                data[i, j] = None
            else:
                x=gdf.CentrX[i]-gdf.CentrX[j]
                y=gdf.CentrY[i]-gdf.CentrY[j]
                dat=hypot(x,y)
                data[i,j]=dat
                res.append(dat)
    return data, res

def cntr_xy_stat(res, title='', the_stop=False):
    print('\nFunction name = ', inspect.currentframe().f_code.co_name)
    print('Статистика =', title)
    print(f'{min(res)=:.4f}')
    print(f'{max(res)=:.4f}')
    print(f'{statistics.mean(res)=:.4f}')
    print(f'{statistics.median(res)=:.4f}')
    print(f'{statistics.mode(res)=:.4f}')
    print(f'{statistics.quantiles(res)=}')
    counts, bins = np.histogram(res)
    print(f'{counts=}')
    print(f'{bins=}')

    plt.title('Гистограмма расстояний между центрами структур и ПУ')
    plt.hist(res, density=False, bins=10, rwidth=0.95)
    plt.xlabel('Расстояние, км')
    plt.ylabel('Число пар')
    plt.grid()
    plt.show()
    if the_stop: sys.exit()



# Определдение расстояний между центрами, получение матрицы, модельный пример, квадратная сетка
def fun_CentrXY_mod(view=False):
    """
    Определение расстояний между центрами. Попарные расстояния, матрицы, модельный пример, квадратная сетка
    """
    print('Function name = ', inspect.currentframe().f_code.co_name)
    nn=51
    print('================== 1')
    data=[]
    for i in range(nn):
        for j in range(nn):
            data.append([i,j])
    print('--------y--------')
    ndata = np.zeros((nn, nn))
    for i in range(nn):
        for j in range(nn):
            if i==j or i<j:
                ndata[i, j] = None
            else:
                x=data[i][0]-data[j][0]
                y=data[i][1]-data[j][1]
                ndata[i,j]=hypot(x,y)
    print('--------ymatr--------')
    print(ndata)
    return ndata


def view_color_grid(data1):
    n=13
    plt.figure(figsize=(n, n+2))
    plt.pcolor(data1,
               cmap=plt.get_cmap('Spectral', 11))
    plt.xlabel('Номер объекта')
    plt.ylabel('Номер объекта')
    plt.colorbar()
    plt.show()



def tst_fun_CentrXY_01nn():
    print('Function name = ', inspect.currentframe().f_code.co_name)
    gpd=input_mifd(UK_f_name)
    res_main, res_all = fun_CentrXY_01nn(gpd, an_dist)
    nears='ближайшие точки окружения'; fars='точки окружения'
    cntr_xy_stat(res_main,nears)
    cntr_xy_stat(res_all,fars)
    print(len(res_main),'   ',len(res_all))

    plt.title('Гистограмма расстояний между центрами структур и ПУ')
    plt.hist(res_all, density=False, bins=10, rwidth=0.8,label=fars)
    plt.hist(res_main, density=False, bins=10, rwidth=0.8,label=nears)
    plt.xlabel('Расстояние, км')
    plt.ylabel('Число пар')
    plt.legend()
    plt.grid()
    plt.show()

def view_default_clucters(gdf, an_dist:dict):
    """
    Просмотр поля точек с размеченными кластерами
    """
    print('Function name = ', inspect.currentframe().f_code.co_name)
    ll=len(gdf)
    plt.figure(figsize=(7,11))
    for i in range(ll):  # повсем аномалиям
        nan_ = int(gdf.Labels[i])  # текущий номер аномалии
        xan = gdf.CentrX[i]
        yan = gdf.CentrY[i]
        ncol=an_dist[nan_]['cl']-1
        plt.scatter(xan, yan, color=nclust_color[ncol])
    plt.title('Ручное разбиение аномалий на кластеры')
    fs=13;   plt.xlabel('x, км', fontsize=fs);     plt.ylabel('y, км', fontsize=fs)
    plt.grid()
    #################### Равный шаг по сетке
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ####################
    plt.show()

def tst_view_default_clucters():
    gpd=input_mifd(UK_f_name, view=True)
    view_default_clucters(gpd, an_dist)


# Определение расстояний между центрами аномалий (учитывая номера), получение словаря.
def fun_CentrXY_01nn(gdf, an_dist:dict, view=False):
    """
    Определение расстояний между центрами аномалий (учитывая номера), получение словаря.
    """
    print('Function name = ', inspect.currentframe().f_code.co_name)
    ll=len(gdf)

    # keyl=list(an_dist.keys())
    # valuel = list(an_dist.values())
    res_main=[]; res_all=[]
    with open("intro.txt", "w") as file:
        for i in range(ll):  # повсем аномалиям
            nan_ = int(gdf.Labels[i])            # текущий номер аномалии
            # Индексы в mid - строки, индексы в an_dist -  целые
            near_an = list(map(str,an_dist[nan_]['main'])) # аномалии, ближайшие к текущей
            far_an = list(map(str,an_dist[nan_]['add']))  # аномалии, далекие от текущей
            print(i,' -- ', nan_, near_an, far_an)  #
            xan=gdf.CentrX[i]; yan=gdf.CentrY[i]
            file.write(f'{i:2} --  {nan_}\n')
            for j in near_an:  # по всем ближайшим аномалиям
                dat=gdf.loc[gdf['Labels'] == j]
                xc=dat['CentrX']; yc=dat['CentrY']
                x=xan-xc; y=yan-yc
                dat=hypot(x,y)
                res_main.append(dat)
                res_all.append(dat)
                file.write(f'  {j:2}   {dat:7.3f}   km -- main\n')
                # print('----------')
                # print(xan, yan)
                # sys.exit()

            for j in far_an:  # по всем дальним аномалиям
                dat=gdf.loc[gdf['Labels'] == j]
                xc=dat['CentrX']; yc=dat['CentrY']
                x=xan-xc; y=yan-yc
                dat=hypot(x,y)
                res_all.append(dat)
                file.write(f'  {j:2}   {dat:7.3f}   km -- add\n')

    return res_main, res_all



# Определение расстояний между центрами аномалий (НЕ учитывая номера), получение словаря
def fun_CentrXY_01(gdf:geopandas.geodataframe.GeoDataFrame, view=False):
    """
    Определение расстояний между центрами. Попарные расстояния, получение словаря
    """
    print('Function name = ', inspect.currentframe().f_code.co_name)
    ll=len(gdf)
    dd_ini=dict(); dd_cor=dict()
    for i in range(ll):
        for j in range(ll):
            if i!=j:
                if i<j: ii,jj=i,j
                else: ii,jj=j,i
                k=(ii,jj)
                if k not in dd_ini:
                    x=gdf.CentrX[i]-gdf.CentrX[j]
                    y=gdf.CentrY[i]-gdf.CentrY[j]
                    dd_ini[k]=hypot(x,y)
                    ddd=dd_ini[k]-float(gdf.rad[i])-float(gdf.rad[j])
                    if ddd>0: dd_cor[k]=ddd
                    else: dd_cor[k]=dd_ini[k]*0.5
                    if dd_cor[k]<=0: print(f'{k=} Длина <=0')
                else:
                    dd_ini[(i,j)]=dd_ini[k]
                    dd_cor[(i,j)]=dd_cor[k]
    if view:
        pp('------------')
        pp(dd_ini)
        pp('------------')
        pp(dd_cor)
    return dd_ini, dd_cor
    # https://stackoverflow.com/questions/33094509/correct-sizing-of-markers-in-scatter-plot-to-a-radius-r-in-matplotlib

def view_dict_length0(dd_ini:dict):
    """
    scatter - диаграмма расстояний
    """
    print('Function name = ', inspect.currentframe().f_code.co_name)
    #-------------- 1
    keys = np.array(list(dd_ini.keys()))
    size = np.array(list(dd_ini.values()))
    x=keys[:,0]
    y=keys[:,1]
    cmap = 'seismic'
    plt.figure(figsize=(14, 16))
    plt.scatter(x,y,s=size, c=size)
    plt.colorbar()
    plt.grid()
    plt.show()

def view_dict_length(dd_ini:dict,dd_cor:dict):
    """ Диаграмма точка-точка.
     Размер маркера = расстоянию между точками
     """
    print('Function name = ', inspect.currentframe().f_code.co_name)
    #-------------- 1
    keys = np.array(list(dd_ini.keys()))
    size = np.array(list(dd_ini.values()))
    x=keys[:,0]
    y=keys[:,1]
    cmap = 'seismic'
    plt.figure(figsize=(18, 16))
    plt.subplot(1, 2, 1)
    plt.scatter(x,y,s=size, c=size)
    plt.colorbar()
    plt.grid()
    #-------------- 2
    plt.subplot(1, 2, 2)
    keys = np.array(list(dd_cor.keys()))
    size = np.array(list(dd_cor.values()))
    x=keys[:,0]
    y=keys[:,1]
    plt.scatter(x,y,s=size, c=size)
    plt.colorbar()
    plt.grid()
    # -----------------
    plt.show()

def view_dict_length2(dd:dict):
    """
    Вычисление и визуализация нарастающих расстояний или приращений расстояний
    """
    print('Function name = ', inspect.currentframe().f_code.co_name)
    keys_ = np.array(tuple(dd.keys()))
    values_=np.array(tuple(dd.values()))
    ll=len(keys_)
    print(f'{ll=}')
    print(keys_)
    print(keys_[:,0])
    s=list(set(keys_[:,0]))
    lls=len(s)
    plt.figure(figsize=(16, 16))
    aver=np.zeros(lls-1)
    aver_add=np.zeros(lls-2)
    print(lls, s)
    for ss in s:
        if ss<70:
            lst=[]
            for i in range(ll):
                if keys_[i,0]==ss:
                    # lst.append((values_[i],keys_[i]))
                    lst.append(values_[i]) # +ss*20
            lst.sort()
            lst_add=[lst[j]-lst[j-1] for j in range(1,len(lst))]
            aver += np.array(lst)
            aver_add += np.array(lst_add)
            # plt.plot(range(len(lst)),lst)
            # plt.scatter(range(len(lst)), lst, s=5)
            plt.plot(range(len(lst_add)),lst_add)
            plt.scatter(range(len(lst_add)), lst_add, s=5)
    # plt.plot(range(len(aver)), aver/(lls-1), c='r', linewidth='4')
    plt.plot(range(len(aver_add)), aver_add / (lls-2), c='r', linewidth='4')
    print(f'{np.min(aver_add / (lls-2))=} ')
    print(f'{np.argmin(aver_add/(lls-2))=} ')
    ax = plt.gca()
    ax.set_ylim([0, 2])
    # plt.legend()
    plt.grid()
    plt.show()




def input_mifd(fl_name, sq_name='', view=False)->geopandas.geodataframe.GeoDataFrame:
    print('\nFunction name = ', inspect.currentframe().f_code.co_name)
    path = fl_name+'.mif'
    df = geopandas.read_file(path, encoding='UTF-8')
    # stackoverflow.com/questions/45393123/adding-calculated-column-in-pandas
    df['pdivs'] = df.Perimetr / df.Square
    df['rad'] = np.sqrt(df.Square/pi)
    #--------------- добавление столбца с моим номером кластера
    ncl = add_nclust()
    df['nclucter'] = ncl[:]
    #--------------- добавление столбца-заготовки для нового номера кластера
    df['nclucter_new'] = ncl[:]
    #---------------
    # print(df['pdivs'])
    df_geom=df['geometry']
    gdf = geopandas.GeoDataFrame(df, geometry='geometry') # , crs='epsg:4326'

    if view:
        print(gdf)
        gdf.plot(figsize=(7,11), column='nclucter') # , alpha=0.75
        plt.title(f'Ручное разбиение на кластеры')
        #################### Равный шаг по сетке
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ####################
        plt.grid()
        plt.show()
        # for i in range(1):  # len(gdf)
        #     print(i, df.Labels[i])
        #     print(i, df.geometry[i])
        # print('-- cycle -- geometry')
        # for i in range(len(gdf)):
        #     print(i, gdf.geometry[i])
        print('\n-------------- df')
        print(df)
        print('\n-------------- df_geom')
        print(df_geom)
        print('\n-------------- gdf')
        print(gdf)
        print('\n-------------- gdf.geometry')
        print(gdf.geometry)
        print('\n-------------- gdf.Labels')
        print(gdf.Labels    )
        print('\n-------------- CentrX CentrY - (beg)')
        for i in range(len(gdf)):
            print(f'{i:3} {gdf.CentrX[i]:8.4f} {gdf.CentrY[i]:8.4f}')
        print('-------------- CentrX CentrY - (end)\n')
    return gdf

from geopandas import GeoDataFrame
def work_with_mifd2(fl_name, sq_name=''):
    print(sq_name)
    # print('------- from_file')
    df = GeoDataFrame.from_file(fl_name+'.mid')
    # print('------- view_geoDataFrame')
    # plot_geoDataFrame(df)
    view_geoDataFrame(df)

def plot_geoDataFrame(df):
    print('Function name = ', inspect.currentframe().f_code.co_name)
    # Вся фигура
    # df.plot(figsize=(10,10), edgecolor='purple', facecolor='lightgreen')
    # Только граница
    df.boundary.plot(figsize=(10,10), edgecolor='purple')
    xc=np.array(df.CentrX)
    yc=np.array(df.CentrY)
    # Геом.центр
    plt.scatter(xc,yc, s=3) # s=3 - размер точки
    plt.grid(); plt.show()
    sys.exit()

def view_geoDataFrame(df):
    # https://habr.com/ru/articles/680100
    # Данные
    # https://github.com/datameet/Municipal_Spatial_Data/tree/master/Kolkata
    # https://dnmtechs.com/extracting-points-coordinates-from-a-polygon-in-shapely/
    print('Function name = ', inspect.currentframe().f_code.co_name)
    print(f'{df.shape=}')
    # print(f'{df.head=}')
    print(df.columns)
    # print(df.Perimetr)
    print(f'{type(df)=}')
    # for i in range(len(df)):
    #     print(f'{i:3}  {df.Labels[i]:3}  {df.Perimetr[i]:8.5f}   {df.Square[i]:8.5f}   {df.CentrX[i]:8.5f}    {df.CentrY[i]:8.5f}' )
    # for i in range(len(df)):
    #     print(f'{i:3}  {df.s['geometry'].tolist():3} ') #
    # ---------------
    # gmtol=df.geometry.tolist()
    # # print(df.geometry.tolist())
    # for i in range(len(df)):
    #     print(f'{i:3}  {list(gmtol[i].exterior.coords)}')
    # ---------------
    for i in range(len(df)):
        print(f'{i:3}  {len(list(df.geometry[i].exterior.coords)):4} {list(df.geometry[i].exterior.coords)}')

print('\npspat_work')

if __name__=="__main__":
    # work_with_mifd2(UK_f_name,UK_sq_name)
    # work_with_mifd2('Str_w', UK_sq_name)
    # fun_CentrXY_mod()
    # tst_fun_CentrXY_01nn()
    # tst_view_default_clucters()
    pass

