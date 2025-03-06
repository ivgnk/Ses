"""
Обработка введенных данных

en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset
en.wikipedia.org/wiki/List_of_map_projections
en.wikipedia.org/wiki/SK-42_reference_system
"""
from pspat_const import *
import geopandas

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
    df = GeoDataFrame.from_file(fl_name+'.mid')
    print(df)
    print(df.columns)

if __name__=="__main__":
    # work_with_mifd2(UK_f_name,UK_sq_name)
    work_with_mifd2('Str_w', UK_sq_name)




