import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)
patches_list = []
color_list = []
patches_list.append(matplotlib.patches.Rectangle((-200,-100), 400, 200))
color_list.append('lightgreen')
# patches_list.append(matplotlib.patches.Rectangle((0,150), 300, 20))
# color_list.append('red')
# patches_list.append(matplotlib.patches.Rectangle((-300,-50), 40, 200))
# color_list.append('#0099FF')
# patches_list.append(matplotlib.patches.Circle((-200,-250), radius=90))
# color_list.append('#EB70AA')

our_cmap = ListedColormap(color_list)
patches_collection = PatchCollection(patches_list, cmap=our_cmap)
patches_collection.set_array(np.arange(len(patches_list)))
ax.add_collection(patches_collection)

plt.xlim([-400, 400])
plt.ylim([-400, 400])
plt.show()