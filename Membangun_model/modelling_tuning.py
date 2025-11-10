import mlflow
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt
import os

# Gunakan MLflow Tracking UI dari DagsHub
mlflow.set_tracking_uri("https://dagshub.com/RiyZ411/msml-studi-kasus-heart.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "RiyZ411"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "d1c842b98e0943929141e4b5b540d7c474fe69c9"

# Data
X_train = pd.read_csv('heart_preprocessing/X_train.csv')
X_test = pd.read_csv('heart_preprocessing/X_test.csv')
y_train = pd.read_csv('heart_preprocessing/y_train.csv').squeeze()
y_test = pd.read_csv('heart_preprocessing/y_test.csv').squeeze()

# Hyperparameter tuning
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, None]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Mulai logging
with mlflow.start_run(run_name="Tuning RF Manual Logging"):
    # Logging parameter terbaik
    for param_name, param_val in grid_search.best_params_.items():
        mlflow.log_param(param_name, param_val)

    # Prediksi
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    # Logging metrik utama
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="macro"))

    # Logging metrik tambahan
    mlflow.log_metric("cohen_kappa", cohen_kappa_score(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        mlflow.log_metric("roc_auc", auc)
    except:
        print("ROC AUC tidak bisa dihitung")

    # Logging confusion matrix sebagai artefak
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("conf_matrix_tuning.png")
    mlflow.log_artifact("conf_matrix_tuning.png")

    # Logging classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    with open("classification_report_tuning.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact("classification_report_tuning.txt")

    # Logging metric_info.json
    metric_info = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1_score": f1_score(y_test, y_pred, average="macro"),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred)
    }
    if 'auc' in locals():
        metric_info["roc_auc"] = auc
    with open("metric_info.json", "w") as f:
        json.dump(metric_info, f, indent=4)
    mlflow.log_artifact("metric_info.json")

    # Logging model
    mlflow.sklearn.log_model(best_model, "best_model_tuning")
