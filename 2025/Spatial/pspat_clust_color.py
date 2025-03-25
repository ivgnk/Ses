"""
    1) Построение модельных распределений
    2) Кластеризация по Поле2D_01_(2024H2).ppt, слайд 86-87
    3) Example 2: Determining the Optimal Number of Clusters
    https://www.slingacademy.com/article/scipy-cluster-vq-kmeans-function-explained-with-examples/
"""
import sys
from random import randint as rit, choice
from random import sample, seed, random, gauss
import inspect
from statistics import median

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from scipy.cluster.vq import kmeans, vq

from pspat_work import *

# как разбивать окно в зависмости от числа рисунков
num_sub=dict({1:(1,1),  2:(1,2),  3:(2,2),   4:(2,2),   5:(2,3),
              6:(2,3),  7:(2,4),  8:(2,4),   9:(2,5),  10:(2,5),
              11:(3,4), 12:(3,4), 13:(3,5), 14:(3,5),  15:(3,5),
              16:(4,4), 17:(3,6), 18:(3,6), 19:(4,5),  20:(4,5),
              21:(3,7), 22:(3,8), 23:(3,8), 24:(3,8),  25:(5,5),
              })

max_a=60 # максимальное число аномалий на площади
nseed=0
nmainsub=[2,3,4,5,6,7,8,9,10]
# Онсновной набор цветов
col_main=['darkgray', 'lightgray',  'indianred',   'brown',       'red',
         'coral',    'chocolate',  'peru',        'orange',      'gold',
         'olive',    'yellow',     'yellowgreen', 'lightgreen',  'darkgreen',
         'lime',     'aguamarine', 'darkcyan',    'cyan',        'cadetblue',
         'lightblue','steelblue',  'royalblue',   'navy',        'blue',
         'indigo',   'purple',     'magenta',     'deeppink' ,   'crimson']
nm_col = dict({1:('red'),  2:('red','blue'),
               3:('red','blue','lightgreen'),
               4:('red','blue','lightgreen','yellow'),
               5:('red','blue','lightgreen','yellow', 'magenta'),
               6:('red','blue','lightgreen','yellow', 'magenta', 'brown'),
               7:('red','blue','lightgreen','yellow', 'magenta', 'brown', 'orange'),
               8:('red','blue','lightgreen','yellow', 'magenta', 'brown', 'orange', 'darkgreen'),
               9:('red','blue','lightgreen','yellow', 'magenta', 'brown', 'orange', 'darkgreen', 'cyan'),
              10:('red','blue','lightgreen','yellow', 'magenta', 'brown', 'orange', 'darkgreen', 'cyan', 'pink'),
              11:('red','blue','lightgreen','yellow', 'magenta', 'brown', 'orange', 'darkgreen', 'cyan', 'pink', 'gray')
              })


def make_data_1(k = 4):    # число кластеров
    # k = 4  # число кластеров
    nn=[151, 112, 72, 165]  # число точек в кластере
    XY=((1,1), (2, 5) , (4, 1.5), (5, 4.5)) # центры кластеров
    dxy=((-1,1), (-1,1), (-1,1), (-1,1))  # low-high
    res=np.zeros((1,2)) # для удобства объединения в цикле, первая строка пустая
    for i in range(k):
        np.random.seed(i)
        x = np.random.uniform(dxy[i][0], dxy[i][1], nn[i])
        y = np.random.uniform(dxy[i][0], dxy[i][1], nn[i])
        x+=XY[i][0]
        y+=XY[i][1]
        res = np.vstack((res, np.array((x, y)).T))
    X = res[1:,:] # удаление лишней первой строки
    print(type(X), X.shape)  # <class 'numpy.ndarray'> (500, 2)
    return X, k

def make_gr_cent(k):
    """
    k - число групп
    """
    rk=range(k)
    rk10main = range(0,k*10,9)
    nn = [max_a//k for _ in rk]  # число точек в кластере
    xc = [i for i in rk10main]  # большая последовательность для выборки Х
    yc = [i for i in rk10main]  # большая последовательность для выборки Y
    smpl_x = sample(xc, k)  # выборка Х
    smpl_y = sample(yc, k)  # выборка Y
    XY=[[smpl_x[i], smpl_y[i]] for i in rk]
    print(XY)
    return rk, nn, XY

def make_data_2(k):
    """
    k - число групп
    return: распределение точек по группам, число групп
    """
    seed(nseed)
    #---1----(beg) центры кластеров
    rk, nn, XY =make_gr_cent(k)
    # ---1----(end) центры кластеров
    dxy = [(-k*2,k*2) for i in rk]  # low-high - размах Х и У в кластере
    # Точки в кластере
    res=np.zeros((1,2)) # для удобства объединения в цикле, первая строка пустая
    for i in range(k):
        np.random.seed(i)
        # x=np.zeros(nn[i]);  y=np.zeros(nn[i])
        # for j in range(nn[i]):
        #     xr = np.random.uniform(dxy[i][0], dxy[i][1])
        #     yr = np.random.uniform(dxy[i][0], dxy[i][1])

        x = np.random.uniform(dxy[i][0], dxy[i][1], nn[i])
        y = np.random.uniform(dxy[i][0], dxy[i][1], nn[i])
        x+=XY[i][0]
        y+=XY[i][1]
        plt.scatter(x,y, color=nm_col[k][i], s=20)
        res = np.vstack((res, np.array((x, y)).T))
    X = res[1:,:] # удаление лишней первой строки
    plt.title(str(nseed))
    plt.grid()
    ####################
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ####################
    plt.show()
    return X, k

def tst_seed(ngr=5):
    """
    Выбор лучшей конфигурации для тестового набора точек
    ngr - число групп
    """
    global nseed
    # nseed = 125;  make_data_2(5)
    for i in range(200):
        nseed=i
        make_data_2(ngr)


def the_kmeans3(fun_make_data, k): # функция и число кластеров
    tit = f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    X, k1 = fun_make_data(k)
    centroids, distortion = kmeans(X, k)
    code, data = vq(X, centroids)
    if k==4: colors = ['r', 'g', 'b', 'y']
    else: colors = col_main
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    plt.figure(figsize=(12,12))
    for i in range(k):
        # выбрать только данные наблюдений с меткой кластера == i
        ds = X[np.where(code == i)]
        # наблюдения данных
        plt.scatter(ds[:,0], ds[:,1], c=colors[i],label=str(i))
        # центроиды
        plt.plot(centroids[:,0], centroids[:,1], linestyle=' ', color='k', marker='+', markersize=16)
    plt.title(tit); plt.grid()
    plt.legend(loc="upper left", bbox_to_anchor=(1,1)); plt.show()

def the_kmeans4(fun_make_data, k:int, minc:int, maxc:int): # функция, число групп, мин и макс число кластеров
    """
    :param fun_make_data: функция
    :param k: число исходных групп точек
    :param minc: минимальное число кластеров
    :param maxc: максимальное число кластеров
    :return: None
    """
    tit = f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    X, k1 = fun_make_data(k)
    npart=maxc-minc+1
    r=num_sub[npart][0];  c=num_sub[npart][1]
    # print(npart, r, c)

    plt.figure(figsize=(12,12))
    for j in range(minc, maxc+1):
        centroids, distortion = kmeans(X, j)
        code, data = vq(X, centroids)
        colors = nm_col[j]
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
        plt.subplot(r, c, j-minc+1)
        k=j
        for i in range(k):
            # выбрать только данные наблюдений с меткой кластера == i
            ds = X[np.where(code == i)]
            # наблюдения данных
            plt.scatter(ds[:,0], ds[:,1], c=colors[i], label=str(i))
            # центроиды
            plt.plot(centroids[:,0], centroids[:,1], linestyle=' ', color='k', marker='+', markersize=16)
        plt.title(f'Число кластеров={j}'); plt.grid()
        ####################
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ####################
    # plt.legend(loc="upper left", bbox_to_anchor=(1,1));
    plt.show()

# def assin_new_cluster(ini_data, code, gdf):
#     """
#     Назначение из code(book) в gdf['nclucter_new']
#     """
#     print('Function name = ', inspect.currentframe().f_code.co_name)
#     ll=len(gdf)
#     plt.figure(figsize=(12, 12))
#     for i in range(ll):
#         ds = ini_data[np.where(code == i)]
#         # plt.scatter(ds[:, 0], ds[:, 1], c=colors[i], label=str(i))
#     ####################
#     ax = plt.gca()
#     ax.set_aspect('equal', adjustable='box')
#     ####################
#     plt.grid();  plt.show()
#     return gdf

def assign_nclucter_new(gpd, code):
    print('\nFunction name = ', inspect.currentframe().f_code.co_name)
    for i in range(len(code)):
        print(i, code[i], gpd.Labels[i],nclust_color[code[i]])
        gpd.at[i,'nclucter_new']=nclust_color[code[i]]
    print('#########################')
    return gpd


def cluct_kmeans_uk_seism_an(gpd, minc:int, maxc:int, viewc=False): # данные, мин и макс число кластеров)
    ini_data=np.array(gpd[['CentrX','CentrY']])
    print('################ ini_data (beg)')
    print(ini_data)
    print('################ ini_data (end)')

    tit = f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    npart=maxc-minc+1
    r=num_sub[npart][0];  c=num_sub[npart][1]
    # plt.figure(figsize=(12,12))
    for j in range(minc, maxc+1): # цикл по числу кластеров
        centroids, distortion = kmeans(ini_data, j)
        code, data = vq(ini_data, centroids)
        print('### gpd[[''nclucter_new'']] -- 1')
        print(gpd[['Labels', 'nclucter_new']])
        gpd=assign_nclucter_new(gpd, code)
        print('### gpd[[''nclucter_new'']] -- 2')
        print(gpd[['Labels', 'nclucter_new']])
        print(gpd)
        #------ Аномалии с закраской
        gpd.plot(figsize=(7, 11), column='nclucter_new', alpha=1)  #
        # plt.show()
        # sys.exit()
        #################### Равный шаг по сетке
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        # plt.show()
        ####################
        print('################ code (beg)')
        print(len(code))
        print('code')
        print(code)
        print('data')
        print(data)
        print('ini_data')
        print(ini_data)
        print('################ code (end)')
        # colors = nm_col[j]
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
        plt.subplot(r, c, j-minc+1)
        k=j
        for i in range(k):  # цикл по полученному числу кластеров
            # выбрать только данные наблюдений с меткой кластера == i
            ds = ini_data[np.where(code == i)]
            # наблюдения данных
            # plt.scatter(ds[:,0], ds[:,1], label=str(i), edgecolor='black')  # c=nclust_color[i],
            # центроиды
            if viewc:
                plt.plot(centroids[:,0], centroids[:,1], linestyle=' ', color='k', marker='+', markersize=16)
        plt.title(f'Число кластеров={j}'); plt.grid()
        ####################
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ####################
    # plt.legend(loc="upper left", bbox_to_anchor=(1,1));
    plt.show()




def tst_cluct_kmeans_uk_seism_an():
    gpd = input_mifd(UK_f_name, view=bool(1))
    # print(data)
    # cluct_kmeans_uk_seism_an(data, 2, 11)
    nn=11
    cluct_kmeans_uk_seism_an(gpd, nn, nn)


def tst_randint():
    for i in range(20):
        random_number = rit(1, 10)
        print(i, random_number)

def tst_rnd():
    # Данные
    seed(125)
    nn=1200; nbin=10
    plt.figure(figsize=(15,8))
    #------------ 1
    plt.subplot(1,3,1)
    data = [random() for i in range(nn)]
    plt.hist(data, bins=nbin)
    plt.title('random()')
    #------------ 2
    plt.subplot(1,3,2)
    data2 = [uniform(0,1) for i in range(nn)]
    plt.hist(data2, bins=nbin)
    # Отображение графика
    plt.title('uniform()')
    #------------ 3
    plt.subplot(1,3,3)
    data2 = [gauss(0,3) for i in range(nn)]
    plt.hist(data2, bins=nbin)
    # Отображение графика
    plt.title('gauss()')
    plt.show()

def tst_rnd_gauss():
    seed(125)
    nn=12000; nbin=10
    plt.figure(figsize=(20,7))
    maxsigma=6; nris=maxsigma
    for i in range(1,maxsigma+1):
        plt.subplot(1, nris, i)
        data2 = [gauss(0, i) for j in range(nn)]
        plt.hist(data2, bins=nbin)
        plt.title(str(i))
        plt.xlim((-25,25))
    plt.show()

def tst_rnd_gauss2():
    nn=170
    data = [gauss(0, 1) for i in range(nn)]
    print(f'{min(data)=} \n  {max(data)=} \n   {median(data)=}')


if __name__=="__main__":
    # the_kmeans3(make_data_1, 4)
    # the_kmeans3(make_data_2, 5)
    # tst_rnd_gauss()
    # tst_rnd()
    # tst_seed(10)
    # tst_rnd_gauss()
    # tst_rnd_gauss2()
    # the_kmeans4(make_data_2, 10, 2, 11)
    tst_cluct_kmeans_uk_seism_an()
