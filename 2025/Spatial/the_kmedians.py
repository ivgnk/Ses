"""
https://www.kaggle.com/code/rafaelsaraivacampos/k-medians-clustering
"""

import matplotlib.pyplot as plt
from pyclustering.cluster.kmedians import kmedians
from pspat_tst_data import *

n_samples = 1500
n_samples = 100
K = 3
# cor=['blue','green']
# cor=colors

X, y = all_create_data(npoints=n_samples, view1=True)

initial_cluster_centers = X[np.random.permutation(X.shape[0])[:K],:]
print(f'{initial_cluster_centers=}\n')
kmedians_instance = kmedians(X,initial_cluster_centers)
kmedians_instance.process()

# ss=sss2(n_samples


clusters = kmedians_instance.get_clusters()
y_kmedians = np.zeros([X.shape[0]])
for i in range(K):
    y_kmedians[clusters[i]]=i
    # print(f'{len(clusters[i])=}')
    # print(clusters[i])
ss=sss2(n_samples, clusters) # задание размеров точек
C = np.array(kmedians_instance.get_medians())
ax = plt.axes()
# Центры кластеров

ax.scatter(C[:,0],C[:,1],marker='+',c='black', s=86)
plt.grid()
print(f'{K=}')
for i in range(K):
    idx = np.where(y_kmedians==i)
    # plt.plot(X[idx,0].tolist(),X[idx,1].tolist(), color=colors[i])
    # print(i, type(X[idx,0]), type(X[idx,1]))
    # print(X[idx,0].shape,X[idx,1].shape)
    lli=len(idx)
    print(i,ss[i])
    ax.scatter(X[idx,0],X[idx,1],ss[i],c=colors[i])


plt.title('k-median')
plt.axis('equal')
plt.show()
