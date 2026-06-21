# делаем import необходимых библиотек
import os

import mlflow

# устанавливаем локальное хранилище для наших экспериментов
# хранилище должно быть такое же, как и при запуске сервиса
mlflow.set_tracking_uri('file:./mlflow_experiments_store')

# получаем id эксеримента, который создаётся по умолчанию
# эксперимент по умолчанию называется Default
experiment_id = mlflow.get_experiment_by_name("Default").experiment_id

# залогируем тестовую метрику и артефакт
with mlflow.start_run(run_name='Default', experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    mlflow.log_metric("test_metric", 0)
    mlflow.log_artifact("test_artifact.txt", "test_artifact")

print(f"Run id запуска: {run_id}")