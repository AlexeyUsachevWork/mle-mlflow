import psycopg2 as psycopg
import pandas as pd
import mlflow
import os

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": "rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net",
    "port": "6432",
    "dbname": "playground_mle_20260408_8d08444c1d",
    "user": "mle_20260408_8d08444c1d_freetrack",
    "password": "679f2b36a6aa43b19856d7d50f0d451d",
}
assert all([var_value != "" for var_value in list(postgres_credentials.values())])

connection.update(postgres_credentials)

# определяем название таблицы, в которой хранятся наши данные
TABLE_NAME = "users_churn"


# эта конструкция создаёт контекстное управление для соединения с базой данных
# оператор with гарантирует, что соединение будет корректно закрыто после выполнения всех операций с базой данных
# причём закрыто оно будет даже в случае ошибки при работе с базой данных
# это нужно, чтобы не допустить так называемую "утечку памяти"
with psycopg.connect(**connection) as conn:

# создаём объект курсора для выполнения запросов к базе данных
# с помощью метода execute() выполняется SQL-запрос для выборки данных из таблицы TABLE_NAME
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")

				# извлекаем все строки, полученные в результате выполнения запроса
        data = cur.fetchall()

				# получаем список имён столбцов из объекта курсора
        columns = [col[0] for col in cur.description]

# создаём объект DataFrame из полученных данных и имён столбцов
# это позволяет удобно работать с данными в Python с использованием библиотеки Pandas
df = pd.DataFrame(data, columns=columns)

print(f"Размер нашей таблицы: {df.shape[0]} строк; {df.shape[1]} столбцов")

EXPERIMENT_NAME = "churn_task_fio"
RUN_NAME = "data_check"

mlflow.set_tracking_uri(
    f"postgresql://{postgres_credentials['user']}:{postgres_credentials['password']}"
    f"@{postgres_credentials['host']}:{postgres_credentials['port']}/{postgres_credentials['dbname']}"
)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# Запустим логирование учебных метрик stats и, например, файл, который хранится в конкретной директории. Этот путь сохраним в переменную artifact_path:
stats = {"mae": 2, "r2": 0.82}
artifact_path = "artifacts.txt"
with open(artifact_path, "w") as f:
    f.write("Это содержимое артефакта")

# словарь с параметрами модели
params = {'depth': 10, 'learning_rate': 0.2}

# создаём директорию с дополнительными ресурсами
os.makedirs("additional_resources", exist_ok=True)
with open("additional_resources/data_description.txt", "w") as file:
    file.write("This file contains data description")


with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id # получаем id конкретного запуска внутри MLflow

    mlflow.log_metrics(stats)
    mlflow.log_artifact(artifact_path, "artifacts")
    mlflow.log_dict(params, "model_params.json")
    mlflow.log_artifacts("additional_resources")