from pathlib import Path

import pandas as pd

from src.data.preprocessors import IdEncoder, RatingNormalizer

INPUT_PATH = Path("data/processed/ratings.csv")
OUTPUT_PATH = Path("data/processed/features.csv")


def load_ratings() -> pd.DataFrame:
    """
    Carrega o dataset processado
    """
    return pd.read_csv(INPUT_PATH)


def apply_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica encoding de id e normalizacao de rating
    """
    encoder = IdEncoder()
    normalizer = RatingNormalizer()

    data = encoder.fit_transform(data)
    data = normalizer.fit_transform(data)

    print(f"Usuários únicos: {encoder.n_users}")
    print(f"Itens únicos: {encoder.n_items}")

    return data


def save_features(data: pd.DataFrame) -> None:
    """
    Salva as features processadas
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Features salvas em {OUTPUT_PATH}")


def main() -> None:
    data = load_ratings()
    data = apply_preprocessing(data)
    save_features(data)


if __name__ == "__main__":
    main()
