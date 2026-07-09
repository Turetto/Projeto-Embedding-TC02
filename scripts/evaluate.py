import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

FEATURES_PATH = Path("data/processed/features.csv")
METRICS_PATH = Path("metrics/eval_metrics.json")


def load_features() -> tuple[np.ndarray, np.ndarray]:
    """
    Carregar features e variáveis completas
    """
    data = pd.read_csv(FEATURES_PATH)
    x = data[["user_id", "item_id"]].values
    y = data["rating"].values
    return x, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Metricas de avaliação
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }


def save_metrics(metrics: dict) -> None:
    """
    Salvar metrics em json
    """
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Métricas salvas em {METRICS_PATH}")


def print_metrics(metrics: dict) -> None:
    """
    Visualização das metricas
    """
    print("\n=== Métricas de Avaliação ===")
    for name, value in metrics.items():
        print(f"  {name.upper()}: {value:.4f}")


def main() -> None:
    x, y = load_features()
    y_pred = np.full_like(y, fill_value=y.mean())
    metrics = compute_metrics(y, y_pred)
    print_metrics(metrics)
    save_metrics(metrics)


if __name__ == "__main__":
    main()
