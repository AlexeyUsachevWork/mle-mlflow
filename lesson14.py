import mlflow

import pandas as pd

'''
Получите объект класса MlflowClient с подключением к Tracking Server и Model Registry.
'''

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

tracking_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
registry_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"

client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)

'''
Посмотрим всю информацию об эксперименте — это может быть любой из проведённых экспериментов.
Для этого нужно получить его уникальный номер. Сделайте это с помощью запроса через MLflow Python API.
'''

EXPERIMENT_NAME = "experiment_name"
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

'''
Чтобы получить информацию о всех запусках, можно воспользоваться методом search_runs, который возвращает таблицу формата pandas.DataFrame.
'''

experiment_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
).sort_values(by="start_time", ascending=False)

'''
Выведите в виде таблицы все интересующие метрики модели для «Космолайна», которую вы обучили, исключив строки, где метрик нет. К метрикам добавьте колонки run_id и start_time. Метрики, которые нужны:
err1,
err2,
метрика log loss,
recall,
ROC AUC,
F1-мера,
точность — precision.
'''

runs = experiment_runs[[
    "run_id", "start_time",
    "metrics.err1",
    "metrics.err2",
    "metrics.logloss", "metrics.recall", "metrics.auc", "metrics.f1", "metrics.precision"
]]
