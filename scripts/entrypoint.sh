#!/bin/bash
set -e

echo "=== Baixando dados ==="
python scripts/download_data.py

echo "=== Pré-processando ==="
python scripts/preprocess.py

echo "=== Feature engineering ==="
python scripts/feature_engineering.py

echo "=== Treinando modelos ==="
python scripts/train.py

echo "=== Registrando melhor modelo ==="
python scripts/register_model.py