"""
Обработка введенных данных

en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset
en.wikipedia.org/wiki/List_of_map_projections
en.wikipedia.org/wiki/SK-42_reference_system
"""
import numpy as np
from pspat_const import *
import geopandas
import sys
import inspect
import matplotlib.pyplot as plt

def work_with_mifd(fl_name, sq_name=''):
    path = fl_name+'.mif'
    df = geopandas.read_file(path, encoding='UTF-8')
    df_geom=df['geometry']
    gdf = geopandas.GeoDataFrame(df, geometry='geometry', crs='epsg:4326')

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
    # print('\n-------------- gdf.geometry')
    # print(gdf.geometry)
    print('\n-------------- gdf.Labels')
    print(gdf.Labels    )

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




