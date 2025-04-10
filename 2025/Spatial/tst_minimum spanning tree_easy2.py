"""
minimum spanning tree
https://favtutor.com/blogs/prims-algorithm-python
https://blog.esemi.ru/2013/11/mst-python.html
"""

N=10
INF = 10 **9  # задаём большое значение для инициализации начальных значений расстояний от вершин до дерева
distances = [INF] * N  # инициируем массив расстояний до дерева, N — число вершин графа G
distances[start] = 0  # задаём расстояние до корня дерева start равным нулю
intree = [False] * N  # инициируем массив флагов принадлежности вершин к дереву
weight = 0  # задаём начальное значение суммарного веса рёбер минимального остовного дерева
for i in range(N):
    min_distance = INF  # выберем очередную вершину u с минимальным расстоянием до дерева и добавим её к дереву
    for j in range(N):
        if not intree[j] and distances[j] < min_distance:
            min_distance = distances[j]
            u = j
            weight += min_distance
            intree[u] = True  # отметим вершину u, как добавленную к дереву
            # пересчитаем минимальные расстояния до добавленной в дерево вершины u
    for v in range(N):
        distances[v] = min(distances[v], G[u][v])
