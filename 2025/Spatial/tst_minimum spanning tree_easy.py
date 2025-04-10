"""
minimum spanning tree
https://favtutor.com/blogs/prims-algorithm-python
https://blog.esemi.ru/2013/11/mst-python.html
"""

from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from pspat_rnd_data import *
from pspat_tst_data import *

# https://blog.esemi.ru/2013/11/mst-python.html
from scipy.spatial.distance import pdist
data, names = all_create_data(npoints=100, view1=True)
data = the_rnd_data

dist = pdist(data, 'euclidean')

# Создание исходного графа (s) с набором данных, где вес каждого ребра равен расстоянию близости между объектами
# [1](https://blog.esemi.ru/2013/11/mst-python.html)
s = nx.Graph()
s.add_nodes_from(names)
dq = deque(dist)
len_x = len(names)
for x in range(len_x - 1):
    for y in range(x + 1, len_x):
        s.add_edge(names[x], names[y], weight=dq.popleft())

# Построение минимального покрывающего дерева полученного графа (mst) с помощью метода nx.minimum_spanning_tree(s)
# [1](https://blog.esemi.ru/2013/11/mst-python.html)
mst = nx.minimum_spanning_tree(s)
# plt.hist([edge ['weight'] for edge in mst.edges_iter(data=True)], 100, color='red', alpha=0.3)

r = nx.Graph()
r.add_nodes_from(names)
edges = [edge for edge in mst.edges_iter(data=True) if edge[2]['weight'] <= 0.05]
r.add_edges_from(edges)

def graph_draw(g):
    """Draw graph"""
    plt.figure()
    nx.draw_graphviz(g, with_labels=False, node_size=3, prog='neato')

graph_draw(mst)
graph_draw(r)
plt.show()
