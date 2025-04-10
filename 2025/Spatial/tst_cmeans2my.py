"""
Fuzzy c-means clustering
https://scikit-fuzzy.github.io/scikit-fuzzy/auto_examples/plot_cmeans.html
"""

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from pspat_tst_data import *

dat, metki = all_create_data(npoints=100, view1=True)
xpts=dat[:,0]
ypts=dat[:,1]
labels=metki
# Visualize the test data
# fig0, ax0 = plt.subplots()
# for label in range(3):
#     ax0.plot(xpts[labels == label], ypts[labels == label], '.',
#              color=colors[label])
# ax0.set_title('Test data: 200 points x3 clusters.')

###########################################
# n_samples = 100
# X, y = all_create_data(npoints=n_samples, view1=True)
# xpts=X[:0]
# ypts=X[:1]
#############################################

# plt.figure(figsize=(12, 8))
alldata = np.vstack((xpts, ypts))
fpcs = []

ncenters = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

# Store fpc values for later
fpcs.append(fpc)

# Plot assigned clusters, for each data point in training set
cluster_membership = np.argmax(u, axis=0)
for j in range(ncenters):
    plt.plot(xpts[cluster_membership == j],
            ypts[cluster_membership == j], '.', color=colors[j], markersize=13)

# Mark the center of each fuzzy cluster
for pt in cntr:
    plt.plot(pt[0], pt[1], marker="+", c='k', markersize=15)

plt.title('Метод c-means')
# ax.axis('off')
plt.axis('equal')
plt.grid()
plt.show()