"""
Обработка введенных данных

en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset
en.wikipedia.org/wiki/List_of_map_projections
en.wikipedia.org/wiki/SK-42_reference_system
"""
import numpy as np
from scipy.constants import value

from pspat_const import *
import geopandas
import sys
import inspect
import matplotlib.pyplot as plt
from math import hypot
from pprint import pp
from math import pi


# Определдение расстояний между центрами
def fun_CentrXY_01(gdf:geopandas.geodataframe.GeoDataFrame, view=False):
    """
    Определение расстояний между центрами.
    Попарные расстояния
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
                    dd_cor[k]=dd_ini[k]-float(gdf.rad[i])-float(gdf.rad[j])
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
    Вычисление и изуализация нарастающих расстояний или приращений расстояний
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
    print('Function name = ', inspect.currentframe().f_code.co_name)
    path = fl_name+'.mif'
    df = geopandas.read_file(path, encoding='UTF-8')
    # stackoverflow.com/questions/45393123/adding-calculated-column-in-pandas
    df['pdivs'] = df.Perimetr / df.Square
    df['rad'] = np.sqrt(df.Square/pi)
    print(df['pdivs'])
    df_geom=df['geometry']
    gdf = geopandas.GeoDataFrame(df, geometry='geometry', crs='epsg:4326')

    if view:
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

if __name__=="__main__":
    # work_with_mifd2(UK_f_name,UK_sq_name)
    work_with_mifd2('Str_w', UK_sq_name)




