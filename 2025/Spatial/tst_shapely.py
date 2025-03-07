"""
https://shapely.readthedocs.io/en/stable/manual.html
"""
import sys

import numpy as np
from shapely import *
from shapely.ops import nearest_points
from tst_dat import *
import matplotlib.pyplot as plt

def spes_tst():
    # https://stackoverflow.com/questions/75699024/finding-the-centroid-of-a-polygon-in-python
    dat=[(0, 0), (70, 0), (70, 25), (45, 45), (45, 180), (95, 188), (95, 200),
         (-25, 200), (-25, 188), (25, 180), (25, 45), (0, 25), (0, 0)]
    points = np.array(dat)
    centroid = np.mean(points, axis=0)
    print(centroid)
    plg = Polygon(dat)
    print(f'{plg.centroid=}');  print()
    xx,yy = plg.exterior.coords.xy
    print(xx.tolist())
    print(yy.tolist())


    # Extract points/coordinates from a polygon in Shapely
    # https://stackoverflow.com/questions/20474549/extract-points-coordinates-from-a-polygon-in-shapely


def main_tst():
    plgn0=Polygon(dat0);  plgn1=Polygon(dat1)
    d0n=np.array(dat0); d1n=np.array(dat1)

    print(plgn0.area)
    # min max
    print(plgn0.bounds)  # minx, miny, maxx, maxy
    min_ = np.min(dat0, axis=0)
    max_ = np.max(dat0, axis=0)
    print(f'{min_[0]=} {min_[1]=}    \n{max_[0]=} {max_[1]=}')
    print(f'{plgn0.length=}')
    # Returns the smallest distance by which a node could be moved to produce an invalid geometry
    # print(f'{plgn.minimum_clearance=}')
    # Returns the minimum distance (float) to the other geometric object.
    # print(f'{plgn0.distance(plgn1)=}')
    print(f'{plgn0.centroid=}')
    print(f'{plgn1.centroid=}')

    # mean0_ = np.mean(d0n, axis=0); print(f'{mean0_=}')
    # mean1_ = np.mean(d1n, axis=0); print(f'{mean1_=}')
    pnt=nearest_points(plgn0, plgn1)
    xx1, yy1 = pnt[0].coords.xy
    xx2, yy2 = pnt[1].coords.xy
    print(pnt)
    print(xx1.tolist()[0], yy1.tolist()[0])
    print(xx2.tolist()[0], yy2.tolist()[0])
    np0=list(pnt[0].coords)[0]
    print()
    print(f'{np0=} {np0[0]=} {np0[1]=}')  #
    plt.title('Anom')
    plt.plot(d0n[:, 0], d0n[:, 1])
    plt.plot(d1n[:, 0], d1n[:, 1])
    plt.scatter(xx1.tolist()[0], yy1.tolist()[0])
    plt.scatter(xx2.tolist()[0], yy2.tolist()[0])
    # plt.scatter(np0[0], np0[1], s=64)
    plt.plot(np0[0], np0[1], marker='x', markersize=10)
    plt.grid(); plt.show()

if __name__=="__main__":
    # spes_tst()
    main_tst()