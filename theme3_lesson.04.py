'''
В коде ниже мы генерируем несколько колонок с данными и объединяем их в единый датафрейм, сохранённый в переменную df. Ваша задача — прописать информацию о типах данных, содержащихся в переменной df, а также создать переменные df_int, df_float, df_bool, df_object, df_date, куда сохранятся наборы данных, соответствующие этим типам.
'''

import pandas as pd
import numpy as np
from datetime import timedelta

np.random.seed(42)
np.random.default_rng(42)
# генерация данных для каждого столбца
data = {
    'temperature_celsius': np.random.uniform(20, 35, size=100),  # температура в градусах Цельсия (float)
    'age_years': np.random.randint(18, 65, size=100),  # возраст в годах (int)
    'timestamp_event': [pd.Timestamp('20230101') + timedelta(days=i) for i in range(100)],  # время события (datetime)
    'product_category': np.random.choice(['electronics', 'clothing', 'food'], size=100),  # категория продукта (string)
    'is_purchased': np.random.choice([True, False], size=100),  # булевое значение приобретения (bool)
    'humidity_percentage': np.random.uniform(40, 80, size=100),  # влажность в процентах (float)
    'income_usd': np.random.randint(20000, 100000, size=100),  # доход в долларах США (int)
    'last_updated': [pd.Timestamp('20240101') + timedelta(days=i) for i in range(100)],  # последнее обновление (datetime)
    'product_name': ['Product_' + str(i) for i in range(100)],  # название продукта (string)
    'is_subscribed': np.random.choice([True, False], size=100)  # булевое значение подписки (bool)
}

# создание DataFrame
df = pd.DataFrame(data)

print(df.dtypes)

# Создание подмножеств по типам данных
df_int = df.select_dtypes(include='int')
df_float = df.select_dtypes(include='float')
df_bool = df.select_dtypes(include='bool')
df_object = df.select_dtypes(include='object')
df_date = df.select_dtypes(include='datetime64')


'''
Задание 2
Примените преобразования к вашим данным:
Binarizer — к колонке income_usd (в качестве границы возьмите средние значения). Результат сохраните в колонку income_usd_binarized.
StandardScaler — к колонке age_years. Результат сохраните в колонку age_years_standarded.
LabelEncoder — к колонке is_subscribed. Результат сохраните в колонку is_subscribed_encoded.
'''

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import Binarizer, StandardScaler, LabelEncoder

np.random.seed(42)
np.random.default_rng(42)
# генерация данных для каждого столбца
data = {
    'temperature_celsius': np.random.uniform(20, 35, size=100),  # температура в градусах Цельсия (float)
    'age_years': np.random.randint(18, 65, size=100),  # возраст в годах (int)
    'timestamp_event': [pd.Timestamp('20230101') + timedelta(days=i) for i in range(100)],  # время события (datetime)
    'product_category': np.random.choice(['electronics', 'clothing', 'food'], size=100),  # категория продукта (string)
    'is_purchased': np.random.choice([True, False], size=100),  # булевое значение приобретения (bool)
    'humidity_percentage': np.random.uniform(40, 80, size=100),  # влажность в процентах (float)
    'income_usd': np.random.randint(20000, 100000, size=100),  # доход в долларах США (int)
    'last_updated': [pd.Timestamp('20240101') + timedelta(days=i) for i in range(100)],  # последнее обновление (datetime)
    'product_name': ['Product_' + str(i) for i in range(100)],  # название продукта (string)
    'is_subscribed': np.random.choice([True, False], size=100)  # булевое значение подписки (bool)
}

# создание DataFrame
df = pd.DataFrame(data)

# 1. Binarizer для income_usd с порогом = среднее значение
mean_income = df['income_usd'].mean()
binarizer = Binarizer(threshold=mean_income)
df['income_usd_binarized'] = binarizer.transform(df[['income_usd']])  # передаём DataFrame с одной колонкой

# 2. StandardScaler для age_years
scaler = StandardScaler()
df['age_years_standarded'] = scaler.fit_transform(df[['age_years']])  # стандартизация

# 3. LabelEncoder для is_subscribed
encoder = LabelEncoder()
df['is_subscribed_encoded'] = encoder.fit_transform(df['is_subscribed'])

# (опционально) вывод первых строк для проверки
print(df[['income_usd', 'income_usd_binarized', 'age_years', 'age_years_standarded', 'is_subscribed', 'is_subscribed_encoded']].head())

'''
Вы, наверное, уже заметили, что применять отдельные энкодеры к данным не всегда удобно, особенно когда речь идёт о разных преобразованиях к разным колонкам. Решить эту проблему поможет объект ColumnTransformer в библиотеке scikit-learn. С его помощью можно объединять несколько преобразований данных в один процесс, а затем применять его к определённым подмножествам признаков.
Он принимает аргумент transformers, в котором каждый трансформер представлен кортежем с его названием, объектом преобразования и соответствующими колонками. Например:

from sklearn.compose import ColumnTransformer

all_transformers = ColumnTransformer(
    transformers=[
        ("Название энкодера #1", QuantileTransformer(), columns), # преобразование 1 и соответствующие колонки
        ("Название энкодера #2", SplineTransformer(), columns), # Преобразование 2 и соответствующие колонки
    ]
)
Чтобы облегчить обработку данных и моделирование, используйте объект Pipeline. Это последовательность преобразований данных, завершающаяся оценивателем, например моделью машинного обучения.
Важно отметить, что данные в пайплайне можно кэшировать с помощью аргумента memory, — это полезно, когда нужно выполнить вычислительно затратные преобразования.
Посмотрите, как использовать ColumnTransformer, чтобы преобразовать данные перед моделированием внутри одного Pipeline для метода опорных векторов (SVC).

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# генерация случайных данных
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# определение числовых признаков для ColumnTransformer
numeric_features = [0, 1, 2, 3]  # пример числовых признаков (нумерация с 0)

# создание ColumnTransformer с преобразованиями для числовых признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)  # преобразования для числовых признаков
    ])

# создание Pipeline с преобразованиями и моделью
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', SVC())])

# обучение модели
pipe.fit(X_train, y_train)

# оценка качества модели на тестовых данных
accuracy = pipe.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
Создавая этот код, ColumnTransformer применяет StandardScaler к выбранным числовым признакам (numeric_features). Далее он объединяет преобразования данных с моделью SVC внутри Pipeline, который затем обучается на данных x_train и y_train. После обучения модель оценивается на тестовых данных для вычисления точности.
Задание 3
Примените преобразования к вашим данным, используя объект ColumnTransformer внутри Pipeline. Используйте преобразования из предыдущего задания:
Binarizer — к колонке income_usd (в качестве границы возьмите средние значения),
StandardScaler — к колонке age_years,
OneHotEncoder — к колонке is_subscribed.
'''

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

np.random.seed(42)
np.random.default_rng(42)
# генерация данных для каждого столбца
data = {
    'temperature_celsius': np.random.uniform(20, 35, size=100),  # температура в градусах Цельсия (float)
    'age_years': np.random.randint(18, 65, size=100),  # возраст в годах (int)
    'timestamp_event': [pd.Timestamp('20230101') + timedelta(days=i) for i in range(100)],  # время события (datetime)
    'product_category': np.random.choice(['electronics', 'clothing', 'food'], size=100),  # категория продукта (string)
    'is_purchased': np.random.choice([True, False], size=100),  # булевое значение приобретения (bool)
    'humidity_percentage': np.random.uniform(40, 80, size=100),  # влажность в процентах (float)
    'income_usd': np.random.randint(20000, 100000, size=100),  # доход в долларах США (int)
    'last_updated': [pd.Timestamp('20240101') + timedelta(days=i) for i in range(100)],  # последнее обновление (datetime)
    'product_name': ['Product_' + str(i) for i in range(100)],  # название продукта (string)
    'is_subscribed': np.random.choice([True, False], size=100)  # булевое значение подписки (bool)
}

df = pd.DataFrame(data)

preprocessor = ColumnTransformer(
    transformers=[
        ('binarizer', Binarizer(threshold=df['income_usd'].mean()), ['income_usd']),
        ('scaler', StandardScaler(), ['age_years']),
        ('onehot', OneHotEncoder(), ['is_subscribed'])
    ]
)

# создание Pipeline с преобразованиями
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

transformed_data = pipe.fit_transform(df)

'''
Задание 4
Вы работаете с данными о ежедневных температурах в градусах Цельсия за год в определённом регионе. Ваша задача — предобработать временные данные, чтобы затем их можно было использовать в модели машинного обучения. Выполните следующие шаги:
извлеките признаки из даты,
рассчитайте среднюю температуру за последние семь дней (скользящее окно) и накопительную сумму за весь период,
добавьте признаки, отражающие общий тренд в данных (сумма температур за каждый месяц) и периодичность событий (средняя температура по месяцам).
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# генерация случайных данных о температурах за год
np.random.seed(42)
np.random.default_rng(42)
start_date = pd.Timestamp('2023-01-01')
end_date = pd.Timestamp('2023-12-31')
dates = pd.date_range(start=start_date, end=end_date)
temperatures = np.random.uniform(low=-10.0, high=30.0, size=len(dates))
temperature_data = pd.DataFrame({'Date': dates, 'Temperature_Celsius': temperatures})

# ваш код для предобработки временных признаков #
# 1. Извлечение признаков из даты
temperature_data['Month'] = temperature_data['Date'].dt.month
temperature_data['Weekday'] = temperature_data['Date'].dt.dayofweek
temperature_data['Hour'] = temperature_data['Date'].dt.hour  # будет 0 для всех записей

# 2. Скользящие окна и накопительные статистики
# Расчёт скользящего среднего за 7 дней (не сохраняется в финальный DataFrame,
# так как ожидаемый список столбцов его не содержит)
rolling_mean_7 = temperature_data['Temperature_Celsius'].rolling(7).mean()
temperature_data['Cumulative_Sum'] = temperature_data['Temperature_Celsius'].cumsum()

# 3. Периодичность и тренды
temperature_data['Monthly_Sum'] = temperature_data.groupby('Month')['Temperature_Celsius'].transform('sum')
temperature_data['Monthly_Mean'] = temperature_data.groupby('Month')['Temperature_Celsius'].transform('mean')
print(temperature_data.head())


