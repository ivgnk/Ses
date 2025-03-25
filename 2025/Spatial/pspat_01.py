"""
Ввод данных из inf-файла
"""
# NamedTuple
# https://habr.com/ru/articles/478934/
# from collections import namedtuple
# IniFnStr = namedtuple('IniFnStr', ['age', 'gender', 'name'])

# https://stackoverflow.com/questions/38523965/python-pandas-create-empty-dataframe-specifying-column-dtypes
import numpy as np
import pandas as pd
# from pspat_const import *

import inspect

def import_mid2pandas(fn_mid:str, name_col:list):
    mid = pd.read_csv(fn_mid, header=None)
    mid.columns = name_col
    return mid

def expand_mid(mid):
    xmain = mid[['CentrX', 'CentrY']]
    xcoo = np.array(mid['CentrX'])
    ycoo = np.array(mid['CentrY'])
    lbl = np.array(mid['Labels'])
    # print(xmain)
    # print(mid['Labels'])
    # plt.scatter(mid['CentrX'], mid['CentrY'])
    # plt.grid()
    # plt.show()

    # Номера аномалий
    # for (xi, yi, lbli) in zip(xcoo, ycoo, lbl):
    #     plt.text(xi, yi, lbli) # , va='top', ha='right
    return xmain, xcoo, ycoo, lbl

# https://stackoverflow.com/questions/24284342/insert-a-row-to-pandas-dataframe
def create_empty_pandas2():
    column_names = ["f_name", "sq_name"]
    df = pd.DataFrame(columns=column_names)
    print(f'{df=}')
    print(f'{df.index=}')
    print('-----------')
    for i in range(2):
        df = df._append({'f_name': '++'+str(i), 'sq_name': '++'+str(i+1)}, ignore_index=True)
    print(f'{df=}')

# https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
def input_inf(fname)->pd.DataFrame:
    print('\nFunction name = ',inspect.currentframe().f_code.co_name)
    # df = pd.read_csv(f_inf_nm, delimiter=';')
    # print(df)
    # return pd.read_csv(fname, sep=r';| ')
    return pd.read_csv(fname, delimiter=';')

def view_inf(inf_fn_pd:pd.DataFrame):
    print('Function name = ',inspect.currentframe().f_code.co_name)
    print('1---------')
    print(f'{inf_fn_pd.columns=}')
    print('\n2---------')
    print(f'{list(inf_fn_pd.columns)=}')
    print('\n3---------')
    print(f'inf_fn_pd.iloc[[0]]=\n {inf_fn_pd.iloc[[0]]}')
    print('\n4---------')
    print(list((inf_fn_pd.iloc[[0]]["f_name"]))[0])
    print('\n5---------')
    print(list(inf_fn_pd.iloc[[0]]['sq_name'])[0])
    print('\n6---------')
    print(f'{len(inf_fn_pd)=}')

print('pspat_01')
if __name__=="__main__":
    # print(input_inf(f_inf_nm))
    pass

