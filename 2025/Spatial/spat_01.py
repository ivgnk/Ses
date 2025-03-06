"""
Кластеризация центров грав.аномалий
На основе статьи
Кластеризация пространственных данных – k-means и иерархические алгоритмы
https://cartetika.ru/tpost/uun5jy5tk1-klasterizatsiya-prostranstvennih-dannih
"""

from pspat_const import *
from pspat_01 import *
from pspat_02 import *
from pspat_clust01 import *
from pspat_const import *
from pspat_work import *

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# cluster_for_UKarea:
# mid = import_mid2pandas(UK_mid_fname, name_col)
# xmain, xcoo, ycoo, lbl = expand_mid(mid)
# spat_clust01(xmain)

inf_fn_pd = input_inf(f_inf_nm)
# view_inf(inf_fn_pd)
for i in range(len(inf_fn_pd)):
    sq_name=list(inf_fn_pd.iloc[[0]]['sq_name'])[0]
    print(f'{sq_name=}')
    mifd_fn=list((inf_fn_pd.iloc[[0]]["f_name"]))[0]
    print(f'{mifd_fn=}')
    work_with_mifd(mifd_fn, sq_name)
    # work_with_mifd(fl_name, sq_name)


