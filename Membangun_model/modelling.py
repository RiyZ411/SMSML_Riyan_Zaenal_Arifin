
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Set tracking URI lokal (opsional, default-nya juga lokal)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Aktifkan autologging dari MLflow
mlflow.autolog()

# Load data
X_train = pd.read_csv('heart_preprocessing/X_train.csv')
X_test = pd.read_csv('heart_preprocessing/X_test.csv')
y_train = pd.read_csv('heart_preprocessing/y_train.csv').squeeze()
y_test = pd.read_csv('heart_preprocessing/y_test.csv').squeeze()

# Start MLflow run
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # tampilkan classification report di konsol
    print(classification_report(y_test, y_pred))