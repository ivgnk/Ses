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
clustering = DBSCAN(eps=2.25, min_samples=7).fit(X)
print(f'{clustering.labels_= }')  # array([ 0,  0,  0,  1,  1, -1])
print(f'{clustering= }') # DBSCAN(eps=3, min_samples=2)

plot_clusters(X, clustering.labels_, title='DBSCAN')