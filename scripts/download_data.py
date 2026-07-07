import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path("data/raw")
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_movielens() -> None:
    """
    Baixa e extrai o dataset movielens 100k
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ml-100k.zip"

    print("Baixando MovieLens 100k...")
    try:
        urllib.request.urlretrieve(URL, zip_path)
        print(f"Download concluído: {zip_path} ({zip_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"Erro no download: {e}")
        return

    print("Extraindo...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        zip_path.unlink()
        print(f"Dataset salvo em {DATA_DIR}/ml-100k/")
    except Exception as e:
        print(f"Erro na extração: {e}")


if __name__ == "__main__":
    download_movielens()
