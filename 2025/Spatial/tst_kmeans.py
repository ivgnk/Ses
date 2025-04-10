"""
kmeans
https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html
"""

import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
# features  = np.array([[ 1.9,2.3],
#                       [ 1.5,2.5],
#                       [ 0.8,0.6],
#                       [ 0.4,1.8],
#                       [ 0.1,0.1],
#                       [ 0.2,1.8],
#                       [ 2.0,0.5],
#                       [ 0.3,1.5],
#                       [ 1.0,1.0]])
# whitened = whiten(features)
# book = np.array((whitened[0],whitened[2]))
# kmeans(whitened,book)
#
# codes = 3
# res=kmeans(whitened,codes)
# print(res)

# Create 50 datapoints in two clusters a and b
pts = 50
rng = np.random.default_rng()
a = rng.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
b = rng.multivariate_normal([30, 10],
                            [[10, 2], [2, 1]],
                            size=pts)
features = np.concatenate((a, b))
print('\nfeatures=\n',features)
# Whiten data
whitened = whiten(features)
print('\nwhitened=\n',whitened)
# Find 2 clusters in the data
codebook, distortion = kmeans(whitened, 2)
# Plot whitened data and cluster centers in red
plt.scatter(whitened[:, 0], whitened[:, 1])
plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
plt.grid(); plt.show()