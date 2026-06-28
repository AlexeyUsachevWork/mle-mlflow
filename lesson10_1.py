import os

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

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

# 1. Загружаем данные (пример)
df = pd.read_csv('users_churn.csv')

# Исключаем идентификаторы и целевую переменную
# Удаляем исходные даты и идентификаторы
X = df.drop(['id', 'customer_id', 'begin_date', 'end_date', 'target'], axis=1)
y = df['target']


# Категориальные признаки: все object, кроме дат (мы их удалили)
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# === ОБРАБОТКА ПРОПУСКОВ ===
for col in cat_features:
    X[col] = X[col].fillna('Unknown')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(
    iterations=100,
    random_seed=42,
    cat_features=cat_features,
    verbose=False
)
model.fit(X_train, y_train)

# 4. Получаем предсказания (для сигнатуры)
prediction = model.predict(X_test)
model.save_model('./models/model.cbm')

mlflow.set_tracking_uri(
    f"postgresql://{postgres_credentials['user']}:{postgres_credentials['password']}"
    f"@{postgres_credentials['host']}:{postgres_credentials['port']}/{postgres_credentials['dbname']}"
)

EXPERIMENT_NAME = "churn_experiment_l10_2" # ваш код здесь (напишите своё уникальное имя эксперимента)
RUN_NAME = "model_1_registry"
REGISTRY_MODEL_NAME = "churn_model_l10_2"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')

# Путь до requirements.txt
pip_requirements = './requirements.txt'

# Имя модели для реестра (используйте переменную)
REGISTRY_MODEL_NAME = "CatboostModel"
ARTIFACT_LOCATION = "s3://s3-student-mle-20260408-8d08444c1d-freetrack"

# ------------------------------------------------------------
# ИСПРАВЛЕНИЕ: приводим все целочисленные колонки к float64,
# чтобы сигнатура корректно обрабатывала возможные NaN в будущем
# ------------------------------------------------------------
int_cols = X_test.select_dtypes(include='int').columns
if len(int_cols) > 0:
    X_test_for_signature = X_test.astype({col: 'float64' for col in int_cols})
else:
    X_test_for_signature = X_test


signature = mlflow.models.infer_signature(X_test_for_signature, prediction)
input_example = X_test[:10]
metadata = {"target_name": "churn"}
code_paths = ["train.py", "val_model.py"]
metadata = {'model_type': 'monthly'}

# Теперь можно получить ID активного эксперимента
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=ARTIFACT_LOCATION
    )
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

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