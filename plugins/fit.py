# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
import yaml
import os
import joblib


def fit_model():
    """
    Основная функция обучения модели:
    - читает параметры из params.yaml
    - загружает данные из data/initial_data.csv
    - разделяет признаки на бинарные категориальные, многозначные категориальные и числовые
    - создает пайплайн предобработки и модели CatBoost
    - обучает на всех данных
    - сохраняет полный пайплайн (pkl) и только модель CatBoost (cbm)
    """
    # 1. Чтение гиперпараметров
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # 2. Загрузка данных
    data = pd.read_csv('data/initial_data.csv')

    # 3. Разделение признаков по типам
    cat_features = data.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = data.select_dtypes(include=['float', 'int'])

    # 4. Предобработчик (ColumnTransformer)
    preprocessor = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop='if_binary'), binary_cat_features.columns.tolist()),
            ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    # 5. Создание модели CatBoost с параметрами из yaml
    #    Если секция 'model' отсутствует, используем пустой словарь
    model_params = params.get('model', {})
    # Гарантируем балансировку весов, если она не задана явно
    if 'auto_class_weights' not in model_params:
        model_params['auto_class_weights'] = 'Balanced'
    model = CatBoostClassifier(**model_params)

    # 6. Сборка и обучение пайплайна
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(data, data['target'])

    # 7. Сохранение моделей
    os.makedirs('models', exist_ok=True)

    # 7a. Полный пайплайн (для быстрого применения в Python)
    joblib.dump(pipeline, 'models/fitted_model.pkl')

    # 7b. Только модель CatBoost (нативный формат, меньший размер, кроссплатформенность)
    pipeline.named_steps['model'].save_model('models/fitted_model.cbm')

    print("Модели успешно сохранены:")
    print("  - full pipeline: models/fitted_model.pkl")
    print("  - CatBoost only: models/fitted_model.cbm")


if __name__ == '__main__':
    fit_model()