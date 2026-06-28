'''
скрипт скорее всего не рабочий нужно уточнять  параметры

описание заданий
Прежде чем переходить к работе с MLflow API, убедитесь, что вам доступна сама модель, то есть она зарегистрирована в реестре моделей MLflow, и поднят MLflow. Если это не так, то:
Поднимите MLflow, используя предложенный ранее вариант конфигурации: локальный Tracking Server, удалённое хранилище для экспериментов и хранилище для артефактов.
Добавьте модель в реестр моделей.
Теперь обучите новую версию модели. После этого обновите версию модели в реестре и не забудьте посчитать метрики: ROC-AUC, precision, recall и другие, чтобы сравнивать модели между собой.

Достаньте модель из реестра, а ещё:
Выведите информацию о состояниях модели.
Поменяйте статус последней версии модели на Production, а предпоследней — на Staging.
Переименуйте модель, добавив к её названию приписку, указывающую, что это модель для категории B2C — в коде укажите через _b2c.



'''

import os

import mlflow


EXPERIMENT_NAME = "churn_L13"
RUN_NAME = "model_0_registry"
REGISTRY_MODEL_NAME = "churn_model_L13"


TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('S3_ACCESS_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('S3_SECRET_KEY')


mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")


pip_requirements = './requirements.txt'
signature = mlflow.models.infer_signature(X_test, prediction)
input_example = X_test[:10]
metadata = {'model_type': 'monthly'}


experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    # ваш код логирования метрик здесь
    mlflow.log_metrics(metrics)

    model_info = mlflow.catboost.log_model(
        cb_model=model,                              # обученная модель CatBoost
        artifact_path="models",                      # путь к артефакту
        pip_requirements=pip_requirements,
        signature=signature,
        input_example=input_example,
        await_registration_for=60,
        metadata=metadata,
        code_paths=code_paths,
        registered_model_name=REGISTRY_MODEL_NAME    # используется переменная
    )

client = mlflow.MlflowClient()
'''
или
tracking_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"
registry_uri = f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}"

client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
'''


models = client.search_model_versions(filter_string=f"name = '{REGISTRY_MODEL_NAME}'")
print(f"Model info:\n {models}")

# Заполнение переменных для первой (последней) версии
model_name_1 = models[-1].name
model_version_1 = models[-1].version
model_stage_1 = models[-1].current_stage

# Заполнение переменных для второй (предпоследней) версии
model_name_2 = models[-2].name
model_version_2 = models[-2].version
model_stage_2 = models[-2].current_stage

print(f"Текущий stage модели 1: {model_stage_1}")
print(f"Текущий stage модели 2: {model_stage_2}")

# Изменение статусов версий
client.transition_model_version_stage(model_name_1, model_version_1, 'production')
client.transition_model_version_stage(model_name_2, model_version_2, 'staging')

# Переименование модели (добавление суффикса _b2c)
client.rename_registered_model(name=REGISTRY_MODEL_NAME, new_name=f'{REGISTRY_MODEL_NAME}_b2c')