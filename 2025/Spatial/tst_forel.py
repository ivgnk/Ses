"""
https://github.com/tsolakghukasyan/FOREL-clustering/
https://github.com/tsolakghukasyan/FOREL-clustering/blob/master/forel.py
"""
import matplotlib.pyplot as plt
import numpy as np
from pspat_tst_data import *

def cluster(points, radius, tol=1e-1):
    centroids = []; neighborsl = []
    while len(points) != 0:
        current_point = get_random_point(points)
        neighbors = get_neighbors(current_point, radius, points)
        centroid = get_centroid(neighbors)
        while np.linalg.norm(current_point - centroid) > tol:
            current_point = centroid
            neighbors = get_neighbors(current_point, radius, points)
            centroid = get_centroid(neighbors)
        points = remove_points(neighbors, points)
        centroids.append(current_point)
        neighborsl.append(np.array(neighbors))
    return centroids, neighborsl


def get_neighbors(p, radius, points):
    neighbors = [point for point in points if np.linalg.norm(p - point) < radius]
    return np.array(neighbors)


def get_centroid(points):
    return np.array([np.mean(points[:, 0]), np.mean(points[:, 1])])


def get_random_point(points):
    random_index = np.random.choice(len(points), 1)[0]
    return points[random_index]


def remove_points(subset, points):
    points = [p for p in points if p not in subset]
    return points

def the_main():
    n_samples = 1500
    n_samples = 100
    K = 3
    X, y = all_create_data(npoints=n_samples, view1=True)
    centroids, neighbors = cluster(X, float(5))
    ll=len(centroids); n=0
    for i in range(ll):
        print(i)
        print(centroids[i])
        print(neighbors[i])
        lln=len(neighbors[i])
        print(f'{lln=}')
        n+=lln
    print(f'Всего центров {ll=}')
    print(f'Всего точек {n=}')
    plt.title('Алгоритм FOREL')
    for i in range(ll):
        plt.scatter(centroids[i][0], centroids[i][1], marker='+', c='black', s=86, zorder=10)
    for i in range(ll):
         plt.plot(neighbors[i][:,0], neighbors[i][:,1], c=colors[i], marker=".", markersize=15, linestyle='None')
    plt.grid(); plt.show()

if __name__=='__main__':
    the_main()