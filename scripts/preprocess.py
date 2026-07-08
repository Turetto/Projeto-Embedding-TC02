from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/ml-100k/u.data")
OUTPUT_PATH = Path("data/processed/ratings.csv")


def load_raw_ratings() -> pd.DataFrame:
    """
    Carrega o arquivo bruto dos dados
    """
    return pd.read_csv(
        RAW_PATH, sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )


def clean_ratings(data: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza dos dados brutos (removação de duplicatas e valores nulos)
    """
    return data.drop_duplicates().dropna()


def save_processed(data: pd.DataFrame) -> None:
    """
    Salvar o dataset limpo
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Salvo em {OUTPUT_PATH} - {len(data)} registros")


def main() -> None:
    data = load_raw_ratings()
    data = clean_ratings(data)
    save_processed(data)


if __name__ == "__main__":
    main()
