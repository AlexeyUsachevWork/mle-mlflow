import os
from dotenv import load_dotenv

import psycopg

import pandas as pd
from sklearn.datasets import make_classification

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()


'''
Напишите код, с помощью которого вы получите данные из таблицы users_churn в формате pandas.DataFrame.
'''

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"),
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}

connection.update(postgres_credentials)

with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users_churn")
        data = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
df = pd.DataFrame(data, columns=columns)

print(df)

'''
Перед тем как приступить к изучению данных в pandas, настройте параметры отображения, чтобы было удобно работать с большим объёмом информации. Например, можно установить максимальное количество отображаемых столбцов и строк
'''

pd.options.display.max_columns = 100
pd.options.display.max_rows = 64


'''
При работе с DataFrame в pandas методы head() и tail() станут полезными инструментами для предварительного просмотра начальных и конечных строк данных. Например:
'''

data, target = make_classification(n_samples=100, n_features=5, random_state=42)

columns = [f"feature_{i}" for i in range(data.shape[1])]
df = pd.DataFrame(data, columns=columns)
df['target'] = target

# Один из ключевых методов при проведении предварительного анализа данных — describe(). Он выводит основные статистические характеристики числовых данных: среднее, стандартное отклонение, медиану, минимальное и максимальное значения.
print(df.describe())
print('-'*30)

# первые пять строк
print(df.head())
print('-'*30)

# последние пять строк
print(df.tail())
print('-'*30)

'''
Поиск пропущенных значений — важная часть анализа данных.
Вы можете использовать метод isnull(), чтобы определить пропущенные значения в DataFrame, и метод sum()
для подсчёта количества пропущенных в каждом столбце значений, отсортированных в порядке убывания.
Запустите код, чтобы посмотреть, как это работает.
'''
print(df.isnull().sum().sort_values(ascending=False))
print('-'*30)

'''
Последний, но тоже очень важный метод — dtypes, который позволяет просмотреть типы данных в каждом столбце DataFrame.
'''
df.dtypes
print('-'*30)

'''
Вы аналитик в компании, занимающейся интернет-торговлей.
Ваша задача — проанализировать средний объём продаж в разрезе категорий товаров.
Воспользуйтесь методом groupby для агрегации данных по категориям товаров и получите среднее значение объёма продаж.
К агрегированному датафрейму примените метод reset_index, чтобы обновить индексы.
Затем отсортируйте по убыванию полученный агрегированный датафрейм — сделайте это по колонке со средним значением объёма продаж.
Выведите получившийся датафрейм.
'''

# сгенерируем небольшой набор данных для примера
data = {
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics',
                 'Clothing', 'Electronics', 'Electronics', 'Clothing', 'Electronics',
                 'Clothing', 'Electronics', 'Clothing', 'Electronics', 'Clothing'],
    'Sales': [400, 300, 450, 200, 500, 550, 330, 480, 520, 350, 420, 290, 390, 410, 380]
}

df = pd.DataFrame(data)

x = "Category"
y = "Sales"

aggregated_df = pd.DataFrame(df.groupby(x).agg({y: 'mean'})).reset_index()
sorted_df = aggregated_df.sort_values(y, ascending=False).reset_index(drop=True)
print(sorted_df)
print('-'*30)

S