"""
pspat_tst_data.py
Данные для тестирования
"""
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import inspect
import sys

an=45; bn=30; cn=25
bs=50
# bs=10
deflen=100
ll_all=an+bn+cn
colors=['blue', 'red', 'green', # main
        'cyan', 'magenta','yellow', # add
        'orange', 'brown', 'darkgreen',
        'pink', 'gray']

def sss(n_samples:int):
    """
    Функция задания размеров точек
    """
    if n_samples<=100:
        return [50]*n_samples
    else: return [5]*n_samples

def sss2(n_samples:int, arr: Union[list, np.ndarray]):
    """
    Функция задания размеров точек2 по числу кластеров
    """
    ll=len(arr); usl=n_samples<=100; lst=[]
    for i in range(ll):
        lst.append(tuple([50 if usl else 5 for _ in range(len(arr[i]))]))
    return lst

def plot_clusters(dat:[list, np.ndarray], labels:Union[list, np.ndarray], title=''):
    print('\nFunction = ', inspect.currentframe().f_code.co_name)
    plt.title(title)
    ll=len(labels)
    if ll<=deflen: bs1=50
    else: bs1=5
    if isinstance(dat, list): dats=np.array(dat)
    else: dats=dat
    # print(f'{type(dats)=}')

    x=dats[:,0]; y=dats[:,1]
    if isinstance(labels, list): lbl = np.array(dat)
    else: lbl=labels
    if lbl.min()==1: lbl=lbl-1
    clrs=[' ' for i in range(ll)]
    for i in range(ll):
        if lbl[i] != -1:
            clrs[i]=colors[lbl[i]]
        else:
            clrs[i]='black'
    # clrs=[colors[lbl[i]] for i in range(ll)]
    plt.scatter(x, y, marker='o', c=clrs, sizes=[bs1]*ll)
    plt.axis('equal')
    plt.grid()
    plt.show()


def plot_abc(a,b,c, title='',with_show=False):
    """
    Построение диаграммы исходного набора данных из 3 множеств
    """
    print('\nFunction = ', inspect.currentframe().f_code.co_name)
    plt.title(title)
    if len(a)+len(b)+len(c)<=deflen: bs1=50
    else: bs1=5

    # plt.scatter(a[:, 0], a[:, 1], marker='o', c='blue', sizes=[bs1] * an)
    # plt.scatter(b[:, 0], b[:, 1], marker='o', c='red', sizes=[bs1] * bn)
    # plt.scatter(c[:, 0], c[:, 1], marker='o', c='green', sizes=[bs1] * cn)
    dd=(a,b,c)
    ss=([bs1] * an, [bs1] * bn, [bs1] * cn)
    for i in range(3):
        plt.scatter(dd[i][:, 0], dd[i][:, 1], marker='o', c=colors[i], sizes=ss[i])
    if with_show:
        plt.axis('equal')
        plt.grid(); plt.show()

def create_data(an, bn, cn, the_len:int=100, view=True):
    """
    Создание исходного набора данных из 3 множеств
    """
    print('\nFunction = ', inspect.currentframe().f_code.co_name)
    rng = np.random.default_rng(125)
    a = rng.multivariate_normal([0, 6], [[2, 1], [1, 1.5]], size=an)
    b = rng.multivariate_normal([2, 0], [[1, -1], [-1, 3]], size=bn)
    c = rng.multivariate_normal([6, 4], [[5, 0], [0, 1.2]], size=cn)
    # print('\na=\n', a); print('\nb=\n', b); print('\nc=\n', c)
    if view: plot_abc(a,b,c,'ini', True)
    return rng, a, b, c

def concat_dats(a, b, c)->(np.ndarray, np.ndarray):
    """
    Объединение исходных 3 наборов данных в один
    """
    print('\nFunction = ', inspect.currentframe().f_code.co_name)
    res = np.concatenate((a, b, c))
    # print(res);  print(metki)
    return res

def normir_data(an, bn, cn, the_len=100):
    if the_len !=100:
        c = the_len // deflen
        an = an * c
        bn = bn * c
        cn = cn * c
    return an, bn, cn

def all_create_data(npoints=1500, view1=False)->(np.ndarray, np.ndarray):
    global an, bn, cn
    an1, bn1, cn1 = normir_data(an, bn, cn, npoints)
    # print(f'{an1=}   {bn1=}   {cn1=}')
    rng, a, b, c = create_data(an1, bn1, cn1, view=view1)
    dat = concat_dats(a, b, c)
    alls = [0] * an1 + [1] * bn1 + [2] * cn1
    metki=np.array(alls)
    return dat, metki

if __name__=='__main__':
    # rng, a, b, c = create_data(an, bn, cn, the_len=1500, view=True)
    # dat, metki = concat_dats(a, b, c)
    dat, metki = all_create_data(npoints=100, view1=True)
    # print(f'{dat.shape=}   {metki.shape}')
    # print(f'{an=}   {bn=}   {cn=}')