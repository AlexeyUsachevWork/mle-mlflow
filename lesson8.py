import os
import psycopg2 as psycopg
import pandas as pd
import mlflow
from dotenv import load_dotenv

load_dotenv()

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"),
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}

assert all([var_value != "" for var_value in list(postgres_credentials.values())])

connection.update(postgres_credentials)

# определим название таблицы, в которой хранятся наши данные.
TABLE_NAME = "users_churn"

# эта конструкция создаёт контекстное управление для соединения с базой данных
# оператор with гарантирует, что соединение будет корректно закрыто после выполнения всех операций
# закрыто оно будет даже в случае ошибки, чтобы не допустить "утечку памяти"
with psycopg.connect(**connection) as conn:

# создаёт объект курсора для выполнения запросов к базе данных
# с помощью метода execute() выполняется SQL-запрос для выборки данных из таблицы TABLE_NAME
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")

                # извлекаем все строки, полученные в результате выполнения запроса
        data = cur.fetchall()

                # получает список имён столбцов из объекта курсора
        columns = [col[0] for col in cur.description]

# создаёт объект DataFrame из полученных данных и имён столбцов.
# это позволяет удобно работать с данными в Python, используя библиотеку Pandas.
df = pd.DataFrame(data, columns=columns)

# 1. Название колонок вашего датафрейма запишите в текстовый файл
with open("columns.txt", "w", encoding="utf-8") as fio:
    fio.write(", ".join([
    "customer_id", "begin_date", "end_date", "type",
    "paperless_billing", "payment_method", "monthly_charges",
    "total_charges", "internet_service", "online_security",
    "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "gender", "senior_citizen",
    "partner", "dependents", "multiple_lines", "target"
    ]))

counts_columns = [
    "type", "paperless_billing", "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "gender", "senior_citizen", "partner", "dependents",
    "multiple_lines", "target"
]

stats = {}

for col in counts_columns:
	# посчитайте уникальные значения для колонок, где немного уникальных значений (переменная counts_columns)
    column_stat = df[col].value_counts().to_dict()
    column_stat = {f"{col}_{key}": value for key, value in column_stat.items()}

    # обновите словарь stats
    stats.update(column_stat)

stats["data_length"] = df.shape[0]
stats["monthly_charges_min"] = df["monthly_charges"].min()
stats["monthly_charges_max"] = df["monthly_charges"].max() # посчитайте максимальное значение в колонке
stats["monthly_charges_mean"] = df["monthly_charges"].mean() # посчитайте среднее значение в колонке
stats["monthly_charges_median"] = df["monthly_charges"].median() # посчитайте медианное значение в колонке
stats["total_charges_min"] = df["total_charges"].min() # посчитайте минимальное значение в колонке
stats["total_charges_max"] = df["total_charges"].max() # посчитайте максимальное значение в колонке
stats["total_charges_mean"] = df["total_charges"].mean() # посчитайте среднее значение в колонке
stats["total_charges_median"] = df["total_charges"].median() # посчитайте медианное значение в колонке
stats["unique_customers_number"] = df["customer_id"].nunique() # посчитайте кол-во уникальных id
stats["end_date_nan"] = df["end_date"].isna().sum() # посчитайте кол-во пустых строк в колонке

# задаём название эксперимента и имя запуска для логирования в MLflow

EXPERIMENT_NAME = "churn_fio_unique_L8"
RUN_NAME = "data_check"


mlflow.set_tracking_uri(
    f"postgresql://{postgres_credentials['user']}:{postgres_credentials['password']}"
    f"@{postgres_credentials['host']}:{postgres_credentials['port']}/{postgres_credentials['dbname']}"
)

# создаём новый эксперимент в MLflow с указанным названием
# если эксперимент с таким именем уже существует,
# MLflow возвращает идентификатор существующего эксперимента
experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
#experiment = mlflow.set_experiment(EXPERIMENT_NAME) # ваш код здесь
#experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    # получаем уникальный идентификатор запуска эксперимента
    run_id = run.info.run_id# ваш код здесь

    # логируем метрики эксперимента
    # предполагается, что переменная stats содержит словарь с метриками,
    # объявлять переменную stats не надо,
    # где ключи — это названия метрик, а значения — числовые значения метрик
    mlflow.log_metrics(stats) # ваш код здесь

    # логируем файлы как артефакты эксперимента — 'columns.txt' и 'users_churn.csv'
    df.to_csv('users_churn.csv', index=False)
    mlflow.log_artifact('columns.txt', artifact_path='dataframe') # ваш код здесь
    mlflow.log_artifact('users_churn.csv', artifact_path='dataframe') # ваш код здесь


experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
# получаем данные о запуске эксперимента по его уникальному идентификатору
run = mlflow.get_run(run_id)# ваш код здесь


# проверяем, что статус запуска эксперимента изменён на 'FINISHED'
# это утверждение (assert) можно использовать для автоматической проверки того,
# что эксперимент был завершён успешно
assert run.info.status == 'FINISHED'# ваш код здесь

# удаляем файлы 'columns.txt' и 'users_churn.csv' из файловой системы,
# чтобы очистить рабочую среду после логирования артефактов
#os.remove('columns.txt')
#os.remove('users_churn.csv')