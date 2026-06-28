# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os
import numpy as np

# оценка качества модели
def evaluate_model():
    # прочитайте файл с гиперпараметрами params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # загрузите результат прошлого шага: fitted_model.pkl
    model_file = 'models/fitted_model.pkl'
    with open(model_file, 'rb') as fd:
        model = joblib.load(fd)

    # загрузите результат предыдущего шага: inital_data.csv
    data_file = 'data/initial_data.csv'
    data = pd.read_csv(data_file)

    # реализуйте основную логику шага с использованием прочтённых гиперпараметров
    cv_strategy = StratifiedKFold(n_splits=5)
    cv_res = cross_validate(
        model,
        data,
        data['target'],
        cv=cv_strategy,
        n_jobs=-1,
        scoring=['f1', 'roc_auc']
    )

    results = {
        'fit_time': float(np.mean(cv_res['fit_time'])),
        'score_time': float(np.mean(cv_res['score_time'])),
        'test_f1': float(np.mean(cv_res['test_f1'])),
        'test_roc_auc': float(np.mean(cv_res['test_roc_auc']))
    }

    # сохраните результата кросс-валидации в cv_res.json
    os.makedirs('cv_results', exist_ok=True) # создание директории, если её ещё нет
    cv_res_file = 'cv_results/cv_res.json'
    with open('cv_results/cv_res.json', 'w') as fd:
        json.dump(results, fd, indent=4)

if __name__ == '__main__':
    evaluate_model()