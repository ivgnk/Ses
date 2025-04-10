"""
Дендрограмма
[RUS]
https://sky.pro/wiki/python/ierarhicheskaya-klasterizaciya-osnovy-i-primery/
"""
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
from pspat_rnd_data import *

# Пример данных
data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
data = the_rnd_data


# Вычисление расстояний
Z = sch.linkage(data, method='ward')

# Построение дендрограммы
plt.figure(figsize=(10, 7))
sch.dendrogram(Z)
plt.show()

# Обрезка дендрограммы для получения 2 кластеров
clusters = fcluster(Z, 3, criterion='maxclust')
print(clusters)
print(f'{len(clusters)=}  {len(data)}')