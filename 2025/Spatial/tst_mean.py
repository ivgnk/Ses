"""
Геометрические функции
stackoverflow.com/questions/65142720/calculation-of-geometrical-center-from-the-set-of-xyz-coordinates
"""

import numpy as np

def center(nal):
    a = np.array(nal)
    mean = np.mean(a, axis=0)
    x,y=0,0; ll=len(nal)
    n=ll-1
    for i in range(n):
        x+=nal[i][0]
        y+=nal[i][1]
    return mean[0], mean[1], x/n, y/n

dat= [(405.6617, 389.8525), (405.6107, 389.9035), (405.5687, 390.0305), (405.5517, 390.1325),
      (405.5347, 390.31), (405.5517, 390.437), (405.5852, 390.615), (405.6447, 390.8185),
      (405.7632, 391.106), (405.8222, 391.225), (405.9072, 391.352), (405.9577, 391.4025),
      (406.0512, 391.462), (406.1527, 391.4955), (406.2542, 391.521), (406.3642, 391.5295),
      (406.4747, 391.521), (406.5677, 391.4785), (406.6187, 391.4365), (406.6777, 391.3685),
      (406.7287, 391.267), (406.7542, 391.1905), (406.7882, 391.055), (406.7967, 390.9365),
      (406.7967, 390.818), (406.7797, 390.6485), (406.7457, 390.5215), (406.7037, 390.386),
      (406.6697, 390.3015), (406.6272, 390.2165), (406.5767, 390.1405), (406.5002, 390.0725),
      (406.3987, 389.9965), (406.3142, 389.9455), (406.1867, 389.878), (406.0597, 389.827),
      (405.9497, 389.802), (405.8397, 389.7935), (405.7297, 389.8105), (405.6617, 389.8525)]

print(center(dat))