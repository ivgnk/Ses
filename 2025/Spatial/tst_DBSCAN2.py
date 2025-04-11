"""
https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics

from pspat_tst_data import *
from pspat_rnd_data import *

X = the_rnd_data
clustering = DBSCAN(eps=3, min_samples=2).fit(X)
print(f'{clustering.labels_= }')  # array([ 0,  0,  0,  1,  1, -1])
print(f'{clustering= }') # DBSCAN(eps=3, min_samples=2)

plot_clusters(X, clustering.labels_, title='DBSCAN')
