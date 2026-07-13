import mlflow
from mlflow.tracking import MlflowClient

from src.settings import settings

MODEL_NAME = "ecommerce-recomender"


def get_best_run() -> str:
    """
    Retorno o run_id com o menor rmse
    """
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.rmse >0",
        order_by=["metrics.rmse ASC"],
    )

    best_run = runs[0]
    best_rmse = best_run.data.metrics["rmse"]
    print(f"Melhor run: {best_run.info.run_id}")
    print(f"RMSE: {best_rmse:.4f}")
    return best_run.info.run_id, best_rmse


def register_model(run_id: str) -> int:
    """
    Registra o modelo e retorna versao criada
    """
    model_uri = f"runs:/{run_id}/recommender.pt"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"Modelo registrado - versão: {result.version}")
    return int(result.version)


def promote_to_production(version: int) -> None:
    """
    Colocar o modelo em producao
    """
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"Versão {version} promovida para produção!")


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    run_id, rmse = get_best_run()
    version = register_model(run_id)
    promote_to_production(version)


if __name__ == "__main__":
    main()
