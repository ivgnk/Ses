"""
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
"""
from sklearn.cluster import DBSCAN
import numpy as np
from pspat_tst_data import *
from pspat_rnd_data import *

X = np.array([[1, 2], [2, 2], [2, 3],
              [8, 7], [8, 8], [25, 80]])
X = the_rnd_data
eps=1.75  # eps=2.25, min_samples=7
for eps in [1.5, 1.75, 2.0, 2.25]:
    for smpl in range(1,8):
        clustering = DBSCAN(eps=eps, min_samples=smpl).fit(X)
        print(f'{clustering.labels_= }')  # array([ 0,  0,  0,  1,  1, -1])
        print(f'{clustering= }') # DBSCAN(eps=3, min_samples=2)
        plot_clusters(X, clustering.labels_, title=f'DBSCAN, {eps=} {smpl=}')