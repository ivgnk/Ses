"""
Дендрограмма
https://www.jcchouinard.com/hierarchical-clustering/
https://pythonhint.com/post/1083660240028051/tutorial-for-scipyclusterhierarchy
https://jocelyn-ong.github.io/hierarchical-clustering-in-SciPy/
https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html

[RUS]
https://sky.pro/wiki/python/ierarhicheskaya-klasterizaciya-osnovy-i-primery/
"""
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from pspat_tst_data import *
from pspat_rnd_data import *

import sys


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
distances=['braycurtis', 'canberra', 'chebyshev',
          'cityblock', 'correlation', 'cosine',
          'dice', 'euclidean', 'hamming',
          'jaccard', 'jensenshannon', 'kulczynski1',
          'mahalanobis', 'matching', 'minkowski',
          'rogerstanimoto', 'russellrao', 'seuclidean',
          'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

methods=['single','complete', 'average', 'weighted', 'centroid', 'median','ward']
methods=['ward']

# data = [[1, 2], [3, 4], [5, 6], [7, 8]]
data, metki = all_create_data(npoints=100, view1=True)
data= the_rnd_data
print(f'{len(data)=}')
# print(data)
# sys.exit()
# Вычисление расстояний
# metric='euclidean'
# for i,dats in enumerate(data):
#     print(dats)
for i,met in enumerate(methods):
    print(i,met)
    Z = sch.linkage(data, method=met)
    print(f'{len(Z)=} {type(Z)=} {Z.shape=}')
    # print(Z)
    # sys.exit()


    # Построение дендрограммы
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fclusterdata.html
    nclust = sch.fcluster(Z, t=3, criterion='maxclust')
    print(f'{len(nclust)=} {type(nclust)=} {nclust.shape=}')
    # print(nclust,'\n')

    plt.figure(figsize=(10, 7))
    sch.dendrogram(Z)
    plt.title(f'{i} {met=}')
    plt.show()
    print('nclust')
    print(nclust)
    plot_clusters(data, nclust, title = 'Иерархическая клкстеризация')
    plt.show()
