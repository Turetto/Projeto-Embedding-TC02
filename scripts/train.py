import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.models.mlp import MLPRecommender, MLPRecommenderV2
from src.models.trainer import build_dataloader, train_with_early_stopping
from src.settings import settings

INPUT_PATH = Path("data/processed/features.csv")
METRICS_PATH = Path("metrics/train_metrics.json")
MODEL_PATH = Path("models/recommender.pt")


def load_data() -> tuple:
    """
    Carregar e dividir os dados em treino, validação e texte
    """
    data = pd.read_csv(INPUT_PATH)

    x = data[["user_id", "item_id"]].values
    y = data["rating"].values.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=settings.random_seed
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=settings.random_seed
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def to_tensors(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Converter arrays np para tensores do torch
    """
    return (
        torch.tensor(x[:, 0], dtype=torch.long),
        torch.tensor(x[:, 1], dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculo das metricas de avaliação
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def get_n_users_items(x_train: np.ndarray, x_test: np.ndarray) -> tuple[int, int]:
    """
    retorna numero de usuarios e itens unicos
    """
    n_users = int(max(x_train[:, 0].max(), x_test[:, 0].max())) + 1
    n_items = int(max(x_train[:, 1].max(), x_test[:, 1].max())) + 1
    return n_users, n_items


def train_baseline(x_train, y_train, x_test, y_test) -> None:
    """
    Modelo baseline-dummy com registro no Mlflow
    """
    with mlflow.start_run(run_name="baseline_dummy"):
        model = DummyRegressor(strategy="mean")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics = compute_metrics(y_test, y_pred)

        mlflow.log_param("strategy", "mean")
        mlflow.log_metrics(metrics)

        print(f"Baseline - RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

    return metrics


def train_mlp_v1(x_train, x_val, x_test, y_train, y_val, y_test) -> dict:
    """
    Treinar rede neural e log no mlflow (v1)
    """

    n_users, n_items = get_n_users_items(x_train, x_test)

    with mlflow.start_run(run_name="mlp_v1"):
        mlflow.log_params(
            {
                "model": "mlp_V1",
                "n_users": n_users,
                "n_items": n_items,
                "embedding_dim": 32,
                "lr": 0.001,
                "patience": 5,
                "random_seed": settings.random_seed,
            }
        )

        model = MLPRecommender(n_users=n_users, n_items=n_items, embedding_dim=32)

        train_loader = build_dataloader(*to_tensors(x_train, y_train))
        val_loader = build_dataloader(*to_tensors(x_val, y_val))

        history = train_with_early_stopping(model, train_loader, val_loader)

        for entry in history:
            mlflow.log_metrics(
                {"train_loss": entry["train_loss"], "val_loss": entry["val_loss"]},
                step=entry["epoch"],
            )

        model.eval()
        with torch.no_grad():
            user_t, item_t, _ = to_tensors(x_test, y_test)
            y_pred = model(user_t, item_t).numpy()

        metrics = compute_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        print(f"MLP - RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

    return metrics


def train_mlp_v2(x_train, x_val, x_test, y_train, y_val, y_test) -> dict:
    """
    Treinar rede neural e log no mlflow (v2)
    """

    n_users, n_items = get_n_users_items(x_train, x_test)

    with mlflow.start_run(run_name="mlp_v2"):
        mlflow.log_params(
            {
                "model": "mlp_V2",
                "n_users": n_users,
                "n_items": n_items,
                "embedding_dim": 32,
                "lr": 0.001,
                "patience": 10,
                "batch_norm": True,
                "random_seed": settings.random_seed,
            }
        )

        model = MLPRecommenderV2(n_users=n_users, n_items=n_items, embedding_dim=64)

        train_loader = build_dataloader(*to_tensors(x_train, y_train))
        val_loader = build_dataloader(*to_tensors(x_val, y_val))

        history = train_with_early_stopping(
            model, train_loader, val_loader, weight_decay=1e-4, patience=10, lr=0.001
        )

        for entry in history:
            mlflow.log_metrics(
                {
                    "train_loss": entry["train_loss"],
                    "val_loss": entry["val_loss"],
                    "lr": entry["lr"],
                },
                step=entry["epoch"],
            )

        model.eval()
        with torch.no_grad():
            user_t, item_t, _ = to_tensors(x_test, y_test)
            y_pred = model(user_t, item_t).numpy()

        metrics = compute_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        print(f"MLP - RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(MODEL_PATH))
        mlflow.log_artifact(str(MODEL_PATH))

    return metrics


def main() -> None:
    torch.manual_seed(settings.random_seed)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    x_train, x_val, x_test, y_train, y_val, y_test = load_data()

    train_baseline(x_train, y_train, x_test, y_test)
    train_mlp_v1(x_train, x_val, x_test, y_train, y_val, y_test)
    metrics = train_mlp_v2(x_train, x_val, x_test, y_train, y_val, y_test)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
