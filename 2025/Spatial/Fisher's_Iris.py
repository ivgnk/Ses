"""
Ирисы Фишера
https://ru.wikipedia.org/wiki/Ирисы_Фишера
https://en.wikipedia.org/wiki/Iris_flower_data_set
https://habr.com/ru/companies/nexign/articles/334738/
https://github.com/ASushkov/Iris
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.jupyter import display

from sklearn import datasets
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate
from sklearn import metrics
from pandas import DataFrame

##### ---1--- Сбор и очистка данных
# Загружаем набор данных:
iris = datasets.load_iris()
# Смотрим на названия переменных
print(iris.feature_names)
# Смотрим на данные, выводим 10 первых строк:
print(iris.data[:10])
# Смотрим на целевую переменную:
print(iris.target_names)
print(iris.target)


iris_frame = DataFrame(iris.data)
# Делаем имена колонок такие же, как имена переменных:
iris_frame.columns = iris.feature_names
# Добавляем столбец с целевой переменной:
iris_frame['target'] = iris.target
# Для наглядности добавляем столбец с сортами:
iris_frame['name'] = iris_frame.target.apply(lambda x : iris.target_names[x])
# Смотрим, что получилось:
print(iris_frame)

##### ---2--- Описательные статистики

# Строим гистограммы по каждому признаку:
plt.figure(figsize=(18, 16))  # (20, 28)
plot_number = 0
for feature_name in iris['feature_names']:
    for target_name in iris['target_names']:
        plot_number += 1
        plt.subplot(4, 3, plot_number)
        plt.hist(iris_frame[iris_frame.name == target_name][feature_name])
        plt.title(target_name)
        plt.xlabel('cm')
        plt.ylabel(feature_name[:-4])
# https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)
plt.show()

import seaborn as sns
sns.pairplot(iris_frame[['sepal length (cm)','sepal width (cm)',
                         'petal length (cm)','petal width (cm)',
                         'name']], hue = 'name')
plt.show()

##### ---3--- Зависимость между переменными
#pd.set_option('display.max_rows', None)
data=iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)',
               'petal width (cm)']].corr()
print(data)

import seaborn as sns
corr = iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
     ax = sns.heatmap(corr, mask=mask, square=True, cbar=False, annot=True, linewidths=.5)
plt.show()


##### ---4--- Данные для обучения и тестовые данные
# from sklearn import cross_validation нет функции cross_validation
train_data, test_data, train_labels, test_labels = cross_validate.train_test_split(iris_frame[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']], iris_frame['target'], test_size = 0.3, random_state = 0)
# визуально проверяем, что получившееся разбиение соответствует нашим ожиданиям:
print(train_data)
print(test_data)
print(train_labels)
print(test_labels)