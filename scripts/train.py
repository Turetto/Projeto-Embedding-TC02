import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.settings import settings

INPUT_PATH = Path("data/processed/features.csv")
METRICS_PATH = Path("metrics/train_metrics.json")


def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega features e divide em treino e teste
    """
    data = pd.read_csv(INPUT_PATH)

    x = data[["user_id", "item_id"]].values
    y = data["rating"].values

    return train_test_split(x, y, test_size=0.2, random_state=settings.random_seed)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculo de acuracia do modelo
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_baseline(x_train, y_train, x_test, y_test) -> None:
    """
    Modelo baseline-dummy com registro no Mlflow
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="baseline_dummy"):
        model = DummyRegressor(strategy="mean")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics = evaluate(y_test, y_pred)

        mlflow.log_param("strategy", "mean")
        mlflow.log_metrics(metrics)

        print(f"Baseline - RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    x_train, x_test, y_train, y_test = load_features()
    train_baseline(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
