"""
kmeans2
https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html
"""
import sys
import inspect
from scipy.cluster.vq import kmeans2
# , vq, kmeans, whiten)
import matplotlib.pyplot as plt
import numpy as np
from pspat_tst_data import *

methods=['random', 'points', '++']

def main_part():
    print('\nFunction = ', inspect.currentframe().f_code.co_name)
    rng, a, b, c = create_data(an, bn, cn)

    print('\na=\n', a)
    print('\nb=\n', b)
    # z = np.concatenate((a, b, c))
    z= concat_dats(a, b, c)
    print('\nz=\n', z)
    # rng.shuffle(z)

    for met in methods:
        # rng.shuffle(z)
        # whitened = whiten(z)
        whitened = z
        print(f'{met=}')
        centroid, label = kmeans2(whitened, 3, minit=met)
        print('\ncentroid=\n', centroid)

        counts = np.bincount(label)
        print('\ncounts=\n', counts)

        w0 = z[label == 0]
        w1 = z[label == 1]
        w2 = z[label == 2]
        plot_abc(a, b, c, 'method=' + met, False)
        ms = 3
        plt.plot(w0[:, 0], w0[:, 1], 'ok', label='cluster 0', markersize=ms)  # ok - черный кружок
        plt.plot(w1[:, 0], w1[:, 1], 'oy', label='cluster 1', markersize=ms)  # oy - желтый кружок
        plt.plot(w2[:, 0], w2[:, 1], 'om', label='cluster 2', markersize=ms)  # om - малиновый кружок
        plt.plot(centroid[:, 0], centroid[:, 1], 'k+', label='centroids', markersize=20)
        plt.axis('equal')
        plt.legend(shadow=True)
        plt.grid(); plt.show()

if __name__=='__main__':
    main_part()



