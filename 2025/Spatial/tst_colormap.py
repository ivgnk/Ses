"""
https://matplotlib.org/stable/users/explain/colors/colormaps.html
"""
from matplotlib import colormaps
import matplotlib.pyplot as plt
import matplotlib as mpl

from colorspacious import cspace_converter

import numpy as np

def cmap01():
    col_map=list(colormaps)
    for i in range(len(col_map)):
        print(i,'  ',col_map[i])

def cmap02():
    cmaps = {}

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(category, cmap_list):
        # Create figure and adjust figure height to number of colormaps
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
        fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                            left=0.2, right=0.99)
        axs[0].set_title(f'{category} colormaps', fontsize=14)

        for ax, name in zip(axs, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
            ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                    transform=ax.transAxes)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axs:
            ax.set_axis_off()

        # Save colormap list for later.
        cmaps[category] = cmap_list
        plt.show()

    plot_color_gradients('Perceptually Uniform Sequential',
                         ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
    plot_color_gradients('Sequential',
                         ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])
    plot_color_gradients('Sequential (2)',
                         ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                          'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                          'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'])
    plot_color_gradients('Diverging',
                         ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                          'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                          'berlin', 'managua', 'vanimo'])

if __name__=='__main__':
    cmap02()
