"""
Подгонка параметров ... распределения к данным с помощью библиотеки scipy.stats
Бета-распределение --- Гамма-распределение
Колмогорова-Смирнова --- Коши
Логнормальное --- Накагами
Райса --- Рэлея
Трейси — Видома --- Фишера
подгонка логарифмически нормального распределения к данным с помощью библиотеки scipy.stats.lognorm
https://stackoverflow.com/questions/36795949/python-testing-if-my-data-follows-a-lognormal-distribution
"""

import math
from scipy import stats

import matplotlib.pyplot as plt
from scipy.stats import lognorm
import numpy as np

def calc_lognorm_pdf_scipy_1():
    """
    1 график при s = 0.954
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
    """
    s = 0.954
    mean, var, skew, kurt = lognorm.stats(s, moments='mvsk')
    x = np.linspace(lognorm.ppf(0.01, s),
                    lognorm.ppf(0.99, s), 100)
    print(f'{x=}')
    plt.plot(x, lognorm.pdf(x, s),
            'r-', lw=5, alpha=0.6, label='lognorm pdf')
    plt.xlim([x[0], x[-1]])
    plt.legend(loc='best', frameon=False)
    plt.show()

def calc_lognorm_pdf_scipy_n():
    """
    несколько графиков
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
    """
    plt.figure(figsize=(12,7))
    s = [0.1, 0.3, 0.5, 0.9, 1.5, 3]
    for i, ss in enumerate(s):
        plt.subplot(2, 3, i+1)
        plt.title('s='+str(ss))
        x = np.linspace(0, 4, 100)
        y = lognorm.pdf(x, ss)
        plt.plot(x, y, lw=2, alpha=1, label='s='+str(ss))
        print(f'\n {i} {ss:2.1f}')
        for i in range(10):
            print(f'{x[i]:0.3f}',end=' ')
        print()
        for i in range(10):
            print(f'{y[i]:0.3f}',end=' ')
        print()
        # plt.xlim([x[0], x[-1]])
        # plt.legend(loc='best', frameon=False)
    plt.show()

def calc_row_col():
    for i in range(6):
        rows = i//3+1; cols=i%3+1
        print(i, rows, cols)

def calc_lognorm_pdf():
    """
    вычислить логнормальную pdf
    https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Normal.logpdf.html
    """
    # standard deviation of normal distribution
    sigma = 0.859455801705594
    # mean of normal distribution
    mu = 0.418749176686875
    # hopefully, total is the value where you need the cdf
    total = 37
    frozen_lognorm = stats.lognorm(s=sigma, scale=math.exp(mu))
    frozen_lognorm.cdf(total)  # use whatever function and val



if __name__=="__main__":
    calc_lognorm_pdf_scipy_n()
    # calc_row_col()