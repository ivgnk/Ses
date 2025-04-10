"""
minimum spanning tree
https://favtutor.com/blogs/prims-algorithm-python
https://blog.esemi.ru/2013/11/mst-python.html
"""

from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

# https://blog.esemi.ru/2013/11/mst-python.html
from scipy.spatial.distance import pdist
names, data = get_data()
dist = pdist(data, 'euclidean')

# Создание исходного графа (s) с набором данных, где вес каждого ребра равен расстоянию близости между объектами
# [1](https://blog.esemi.ru/2013/11/mst-python.html)
s = nx.Graph()
s.add_nodes_from(names)
dq = deque(dist)
len_x = len(names)
for x in xrange(len_x - 1):
    for y in xrange(x + 1, len_x):
        s.add_edge(names[x], names[y], weight=dq.popleft())

# Построение минимального покрывающего дерева полученного графа (mst) с помощью метода nx.minimum_spanning_tree(s)
# [1](https://blog.esemi.ru/2013/11/mst-python.html)
mst = nx.minimum_spanning_tree(s)
plt.hist([edge ['weight'] for edge in mst.edges_iter(data=True)], 100, color='red', alpha=0.3)

