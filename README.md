# Projeto Embedding TC02

Sistema de recomendação de produtos baseado em embeddings de usuários e itens,
treinado com PyTorch sobre o dataset MovieLens 100k.

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker e Docker Compose

## Instalação

```bash
# Clone o repositório
git clone <url-do-repo>
cd Projeto-Embedding-TC02

# Instale as dependências
uv sync --all-groups

# Configure o ambiente
cp .env.example .env

# Valide o ambiente
uv run python scripts/validate_env.py
```

## Dados

```bash
# Baixe o dataset
uv run python scripts/download_data.py

# Versione com DVC
uv run dvc add data/raw/ml-100k
```

## Pipeline

```bash
# Rode o pipeline completo
uv run dvc repro
```

Os stages são executados em ordem:
1. `preprocess` — limpeza dos dados brutos
2. `feature_engineering` — encoding de IDs e normalização
3. `train` — treino do baseline e da MLP
4. `evaluate` — métricas finais

## Treinamento manual

```bash
# Suba o MLflow
uv run mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# Treine os modelos
uv run python scripts/train.py

# Registre o melhor modelo
uv run python scripts/register_model.py
```

Acesse o MLflow em `http://localhost:5000`.

## Docker

```bash
# Suba todos os serviços
docker compose up --build
```

## Resultados

| Modelo   | RMSE   |
|----------|--------|
| Baseline | 0.2810 |
| MLP      | 0.2366 |

## Estrutura do projeto
├── src/
│   ├── models/        # BaseRecommender, MLP, Factory
│   ├── data/          # Preprocessadores (Strategy Pattern)
│   ├── config.py      # Caminhos centralizados
│   └── settings.py    # Configurações via .env
├── scripts/           # Pipeline: download, preprocess, train, evaluate
├── data/              # Dados versionados com DVC
├── models/            # Artefatos treinados
├── metrics/           # Métricas em JSON
├── configs/           # Configurações YAML
├── tests/             # Testes automatizados
├── dvc.yaml           # Pipeline reprodutível
├── docker-compose.yml # Orquestração de serviços
└── MODEL_CARD.md      # Documentação do modelo

## Licença
MIT