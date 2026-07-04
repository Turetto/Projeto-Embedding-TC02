from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuracoes do projeto para .env
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ecommerce-recommender"
    random_seed: int = 42
    data_path: str = "data/raw/ratings.csv"
    model_output_path: str = "models/"


settings = Settings()
