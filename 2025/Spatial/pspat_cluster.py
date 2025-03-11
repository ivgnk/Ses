"""
    Кластеризация по
    Поле2D_01_(2024H2).ppt, слайд 73-88
"""
import numpy as np
import matplotlib.pyplot as plt
import geopandas
import inspect

from scipy.cluster.vq import kmeans, whiten


def work_cluster():
    """
    Кластеризация
    Поле2D_01_(2024H2).ppt, слайд 73-88
    """
    pass

def make_data(nn:int, gdf):
    if gdf is None:
        rng = np.random.default_rng(seed=125)
        points = rng.random((nn, 2))
    else:
        pass
    return points

def fun(i,nn, gdf):
    whitened, codebook, distortion, name = 0, 0, 0, 0
    if i==1:
        points = make_data(nn, gdf)
        whitened = points.copy()
        codebook, distortion = kmeans(whitened, 2)
        name='not whiten, k=2'
    if i==2:
        points = make_data(nn, gdf)
        whitened = whiten(points)
        book = np.array((whitened[0], whitened[2]))
        codebook, distortion = kmeans(whitened, book)
        name='whiten, book'
    if i == 3:
        points = make_data(nn, gdf)
        whitened = whiten(points)
        codebook, distortion = kmeans(whitened, 2)
        name = 'whiten, k=2 '
    if i == 4:
        points = make_data(nn, gdf)
        whitened = whiten(points)
        codebook, distortion = kmeans(whitened, 4)
        name = 'whiten, k=4'
    return whitened, codebook, distortion, name

def the_kmeans(gdf=None):
    tit = f"Function = {inspect.currentframe().f_code.co_name}"
    print(tit)  # Вывод имени функции
    plt.figure(figsize=(16, 8))
    nn=50
    plt.suptitle(tit)
    for i in range(4):
        whitened, codebook, distortion, name=fun(i+1, nn, gdf)
        s=f'\nVar {i} - {name}'; print(s)
        print('codebook=\n',codebook)
        print('distortion=',distortion)
        plt.subplot(2, 2, i+1)
        plt.scatter(whitened[:, 0], whitened[:, 1])
        plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
        plt.title(s)
        plt.grid()
    plt.show()

if __name__=="__main__":
    the_kmeans()
