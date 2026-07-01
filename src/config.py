from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "processed"

RANDOM_SEED: int = 42

