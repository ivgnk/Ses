"""
https://matplotlib.org/stable/gallery/statistics/histogram_multihist.html
"""
import random
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

nn=1000
n_bins = 10
random.seed(125)
# ----- VAR 1
# x1=[]
# for i in range(nn):
#     lst=[random.random() for _ in range(3)]
#     x1.append(lst)
# ----- VAR 2
# x1 = [[random.random() for _ in range(3)] for _ in range(nn)]
# x = np.random.randn(nn, 3)
# print(type(x1))
# print(x1)
x1=[random.random() for _ in range(nn)]
x2=[random.random() for _ in range(nn)]
x3=[random.random() for _ in range(nn)]

colors = ['red', 'tan', 'lime']
plt.hist(np.array(x1), n_bins, density=True, histtype='bar', color=colors, label=colors)
plt.legend(prop={'size': 10})
plt.title('bars with legend')
plt.show()