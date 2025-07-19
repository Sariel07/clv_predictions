import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, confusion_matrix
)

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save any Python object using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load any Python object saved with dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Evaluate and compare models using GridSearchCV and multiple classification metrics.
    """
    try:
        report = {}

        for name, model in models.items():
            try:
                print(f"\n🔍 Training model: {name}")
                hyperparams = param.get(name, {})

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=hyperparams,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=1,
                    error_score='raise',
                    refit=True
                )

                # LightGBM handling: ensure dataframe format
                if "LGBM" in name:
                    if not isinstance(X_train, pd.DataFrame):
                        X_train = pd.DataFrame(X_train, columns=[f"f_{i}" for i in range(X_train.shape[1])])
                    if not isinstance(X_test, pd.DataFrame):
                        X_test = pd.DataFrame(X_test, columns=X_train.columns)

                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_

                # Predict
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Metrics
                metrics = {
                    "Train Accuracy": accuracy_score(y_train, y_train_pred),
                    "Test Accuracy": accuracy_score(y_test, y_test_pred),
                    "Train F1 Score": f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
                    "Test F1 Score": f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
                    "Train Precision": precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                    "Test Precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                    "Train Recall": recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                    "Test Recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                    "Train MSE": mean_squared_error(y_train, y_train_pred),
                    "Test MSE": mean_squared_error(y_test, y_test_pred),
                }

                print(f"\n📊 {name} Performance:")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")

                print(f"Confusion Matrix (Test):\n{confusion_matrix(y_test, y_test_pred)}")

                report[name] = metrics["Test Accuracy"]

            except Exception as model_err:
                print(f"⚠️ Skipping model '{name}' due to error: {model_err}")
                continue

        return report

    except Exception as e:
        raise CustomException(e, sys)
