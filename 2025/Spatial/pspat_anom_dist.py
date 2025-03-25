"""
pspat_anom_dist
Юго-Камская площадь: расстояние до ближайших аномалий
"""
import sys
nclust      =[  1,  2,             3,   4,        5,  6,   7,      8,             9,    10,      11,         12,     13,      14]
nclust_color=['r', 'b', 'lightgreen', 'y', 'orange', 'c', 'm', 'gray',  'darkgreen', 'pink', 'olive', 'crimson', 'navy', 'purple']
an_dist=dict({
     1: dict({'main': (6, 22, 24, 28, 38, 47, 53, 54), 'add': (25,), 'cl': 7}),
     2: dict({'main': (1, 25, 26, 29, 30, 47, 54, ), 'add': (), 'cl':4}),
     6: dict({'main': (1, 24), 'add': (22,), 'cl':7}),
     8: dict({'main': (10,11, 12, 13), 'add':(), 'cl':1}),
     9: dict({'main': (10,36), 'add':(), 'cl':1}),

    ## - 10
    10: dict({'main':(8, 9, 36), 'add':(13,),  'cl':1}),
    11: dict({'main': (8, 12), 'add': (10, 14),'cl':1}),
    12: dict({'main': (8, 11, 13, 14), 'add': (),'cl':3}),
    13: dict({'main': (8, 10, 12, 15, 36, 39, ), 'add': (40,),'cl':3}),
    14: dict({'main': (11, 12, 13, 37, ), 'add': (18, 22, 44 ),'cl':3}),
    15: dict({'main': (13, 39), 'add': (10, 16, 17, 36, 40, 45),'cl':3}),
    16: dict({'main': (15, 40, 41, 45), 'add': (17, 39, 42, ),'cl':2}),
    17: dict({'main': (18, 44, 45), 'add': (15, 15, 39),'cl':2}),
    18: dict({'main': (17, 21, 43, 44, 45), 'add': (1, 6, 22, 25, 26, 47, ),'cl':2}),
    19: dict({'main': (20, 21, 42, 43), 'add': (),'cl':2}),

    ## - 20
    20: dict({'main': (19, 41, 42), 'add': () ,'cl':2}),
    21: dict({'main': (19, 43), 'add': (18, 25, 26),'cl':2}),
    22: dict({'main': (1, 6, 24), 'add': (12, 13, 14, 18, 23, 37, 38, 39, 44),'cl':8}),
    23: dict({'main': (24, 27, 48, 49, 50), 'add': (22, 37, 38),'cl':6}),
    24: dict({'main': (6, 50), 'add': (1, 23, 49, 53),'cl':7}),
    25: dict({'main': (1, 2, 18, 21, 26, 43, 47), 'add': (22,),'cl':9}),
    26: dict({'main': (2, 21, 25), 'add': (18, 30, 56),'cl':9}),
    27: dict({'main': (24, 49, 50, 52, 51), 'add': (23, 32)  ,'cl':6}),
    28: dict({'main': (32, 53, 54), 'add': (1, 6, 24, 29, 33),'cl':5}),
    29: dict({'main': (2, 30, 54), 'add': (1, 32, 33, 47, 56),'cl':4}),
    ## - 30
    30: dict({'main': (2, 29), 'add': (33, 56),'cl':4}),
    31: dict({'main': (32, 51, 52, 55, ), 'add': (53,),'cl':5}),
    32: dict({'main': (28, 31, 33, 34, 35, 52, 53, 54, 57), 'add': (),'cl':5}),
    33: dict({'main': (29, 30, 32, 56, ), 'add': (28, 54),'cl':4}),
    34: dict({'main': (32, 35, 57), 'add': (31, 55),'cl':5}),
    35: dict({'main': (32, 34, 57), 'add': (),'cl':5}),
    36: dict({'main': (9, 10), 'add': (13, 15, 40),'cl':1}),
    37: dict({'main': (14, 38), 'add': (11, 12, 13, 22, 23, 39, 44),'cl':3}),
    38: dict({'main': (37,), 'add': (22, 23, 48),'cl':3}),
    39: dict({'main': (13, 15, 17, 44), 'add': (22,),'cl':3}),
    ## - 40
    40: dict({'main': (15, 16, 36, 39), 'add': (17, 45),'cl':3}),
    41: dict({'main': (16, 20, 42), 'add': (18, 45),'cl':2}),
    42: dict({'main': (19, 20, 41, 43), 'add': (18, 45),'cl':2}),
    43: dict({'main': (18, 19, 21, 42, ), 'add': (25, 45),'cl':2}),
    44: dict({'main': (17, 18, 45), 'add': (1, 12, 13, 14, 22, 37, 39),'cl':2}),
    45: dict({'main': (17, 18, 44), 'add': (41, 42, 43, 16),'cl':2}),
    47: dict({'main': (1,), 'add': (2, 18, 22, 25, ),'cl':7}),
    48: dict({'main': (23, 49, 51), 'add': (38, 52),'cl':6}),
    49: dict({'main': (23, 27, 48, 51, 52), 'add': (24, 50),'cl':6}),
    ## - 50
    50: dict({'main': (24,), 'add': (23, 27, 32, 53),'cl':7}),
    51: dict({'main': (27, 48, 49, 52,), 'add': (31, 55),'cl':6}),
    52: dict({'main': (27, 31, 32, 49, 51, ), 'add': (48, 53,),'cl':6}),
    53: dict({'main': (24, 50, 28, 32), 'add': (1, 2, 27, 31, 52, 54),'cl':5}),
    54: dict({'main': (2, 28, 29), 'add': (1, 6, 24, 32, 53),'cl':4}),
    55: dict({'main': (31,), 'add': (32, 34, 51, 52),'cl':5}),
    56: dict({'main': (29, 30, 33, ), 'add': (32, 57),'cl':4}),
    57: dict({'main': (32, 34, 35), 'add': (),'cl':5}),
})

def raspr_an_main_big()->list:
    """
    Собираем расстояния из 'main', длинно
    """
    lst=[]
    for k, v in an_dist.items():
        # print(k,an_dist[k]['main'])
        for ss in an_dist[k]['main']:
            lst.append(ss)
    return lst

def raspr_an_main_big2()->list:
    """
    Перебор точек 'main', самое длинное
    """
    ll=len(an_dist)
    keyl=list(an_dist.keys())
    valuel = list(an_dist.values())
    print(type(keyl), keyl)
    print(valuel)
    for i in range(ll):
        print(i,' ',keyl[i], end=' == ')
        valuell=valuel[i]['main']
        for j in valuell:
            print(j, end=' ')
        print()
    return []


def raspr_an_main()->list:
    """
    Собираем расстояния из 'main', list comprehension
    """
    return [ss for k, v in an_dist.items() for ss in an_dist[k]['main']]

def raspr_an_add()->list:
    """
    Собираем расстояния из 'add'
    """
    return [ss for k, v in an_dist.items() for ss in an_dist[k]['add']]

def raspr_an_all():
    return raspr_an_main()+raspr_an_add()

def compare_lst():
    """
    Сравнение результатов raspr_an_main и raspr_an_main_big
    """
    lmain1=raspr_an_main();        print(lmain1)
    lmain2=raspr_an_main_big();    print(lmain2)
    print(lmain1==lmain2)

def cntr_ncluster():
    n = sum([1 for k, v in an_dist.items() if an_dist[k]['cl'] not in nclust])
    if n==0:
        print('Все аномалии разнесены по кластерам')
        print('min N clust = ',min([an_dist[k]['cl'] for k, v in an_dist.items()]))
        print('max N clust = ',max([an_dist[k]['cl'] for k, v in an_dist.items()]))
    else: print(f'{n} аномалий не разнесены по кластерам')

if __name__=="__main__":
    # compare_lst()
    # print(raspr_an_add())
    # print(raspr_an_main_big2())
    cntr_ncluster()
