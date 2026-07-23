# Projeto Embedding TC02
### Sistema de Recomendação de Produtos com Embeddings e PyTorch

---

## Descrição

Este projeto implementa um sistema de recomendação de produtos para e-commerce baseado no comportamento de navegação dos usuários. O modelo central é uma rede neural MLP (Multilayer Perceptron) com embeddings de usuários e itens, treinada com PyTorch sobre o dataset MovieLens 100k.

O pipeline completo é containerizado com Docker, os dados são versionados com DVC, os experimentos são rastreados no MLflow e o código segue padrões profissionais de clean code com SOLID e design patterns.

---

## Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|---|---|---|
| Python | 3.14+ | Linguagem principal |
| PyTorch | 2.2+ | Rede neural e embeddings |
| Scikit-Learn | 1.4+ | Baseline e métricas |
| MLflow | 2.x | Tracking de experimentos e Model Registry |
| DVC | 3.x | Versionamento de dados e pipeline |
| Pydantic Settings | 2.x | Gerenciamento de configurações via .env |
| Docker + Compose | — | Containerização e orquestração |
| uv | — | Gerenciamento de dependências |
| ruff | 0.4+ | Linting e formatação |
| pre-commit | 3.x | Hooks automáticos de qualidade |

---

## Requisitos

- Python 3.14 ou superior
- [uv](https://docs.astral.sh/uv/) instalado
- Docker e Docker Compose
- Git

---

## Instalação e Configuração

**1. Clone o repositório:**
```bash
git clone <url-do-repositorio>
cd Projeto-Embedding-TC02
```

**2. Instale as dependências:**
```bash
uv sync --all-groups
```

**3. Configure as variáveis de ambiente:**
```bash
cp .env.example .env
```

Variáveis disponíveis no `.env`:

| Variável | Padrão | Descrição |
|---|---|---|
| `MLFLOW_TRACKING_URI` | http://localhost:5000 | URI do servidor MLflow |
| `MLFLOW_EXPERIMENT_NAME` | ecommerce-recommender | Nome do experimento |
| `RANDOM_SEED` | 42 | Seed para reprodutibilidade |
| `DATA_PATH` | data/raw/ratings.csv | Caminho para o dataset |
| `MODEL_OUTPUT_PATH` | models/ | Caminho para salvar modelos |

**4. Valide o ambiente:**
```bash
uv run python scripts/validate_env.py
```

Saída esperada:
```
✅ Todos os pacotes instalados
✅ Arquivo .env encontrado
✅ Configurações carregadas com sucesso
```

---

## Dataset

O projeto usa o **MovieLens 100k** (GroupLens Research) como proxy de dados de e-commerce.

| Atributo | Valor |
|---|---|
| Usuários | 943 |
| Itens | 1.682 |
| Interações | 100.000 |
| Ratings | 1 a 5 |
| Fonte | https://grouplens.org/datasets/movielens/100k/ |

```bash
# Baixar os dados
uv run python scripts/download_data.py

# Versionar com DVC
uv run dvc add data/raw/ml-100k
git add data/raw/ml-100k.dvc
git commit -m "feat: version dataset with DVC"
```

---

## Pipeline DVC

O pipeline é composto por 4 stages definidos no `dvc.yaml`:

```
preprocess → feature_engineering → train → evaluate
```

| Stage | Script | Entrada | Saída |
|---|---|---|---|
| preprocess | scripts/preprocess.py | data/raw/ml-100k/u.data | data/processed/ratings.csv |
| feature_engineering | scripts/feature_engineering.py | ratings.csv | features.csv |
| train | scripts/train.py | features.csv | recommender.pt + train_metrics.json |
| evaluate | scripts/evaluate.py | features.csv | eval_metrics.json |

```bash
# Rodar o pipeline completo
uv run dvc repro

# Verificar status
uv run dvc status
```

> O DVC detecta automaticamente quais stages precisam ser reexecutados com base nos hashes dos arquivos de entrada. Se nada mudou, pula o stage.

---

## Treinamento Manual

**1. Suba o servidor MLflow:**
```bash
uv run mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

**2. Treine os modelos:**
```bash
uv run python scripts/train.py
```

O script treina dois modelos e loga no MLflow:
- `baseline_dummy` — DummyRegressor com estratégia "mean"
- `mlp_embeddings` — MLP com embeddings de usuários e itens

**3. Acesse o MLflow UI:** http://localhost:5000

**4. Registre e promova o melhor modelo:**
```bash
uv run python scripts/register_model.py
```
O script seleciona automaticamente a run com menor RMSE e promove para `Production` no Model Registry.

---

## Arquitetura do Modelo

```
user_id → Embedding(943, 32)  ─┐
                                ├─ concat(64) → Linear(64) → ReLU → Dropout(0.2)
item_id → Embedding(1682, 32) ─┘               → Linear(32) → ReLU
                                                → Linear(1)  → Sigmoid → score
```

| Parâmetro | Valor |
|---|---|
| Embedding dim | 32 |
| Otimizador | Adam |
| Learning rate | 0.001 |
| Loss | MSELoss |
| Early stopping | patience=5 |
| Seed | 42 |

> **Por que embeddings?** Em vez de passar `user_id=42` diretamente para a rede, o embedding transforma esse ID num vetor denso onde cada dimensão captura características latentes do usuário. A rede aprende essas representações durante o treinamento.

---

## Resultados

Avaliação no conjunto de teste (20% dos dados, ~20.000 interações):

| Modelo | RMSE | MAE |
|---|---|---|
| Baseline (mean) | 0.2810 | 0.2355 |
| MLP Embeddings | 0.2366 | — |

A MLP obteve uma **redução de ~16% no RMSE** em relação ao baseline, confirmando que os embeddings capturam padrões reais de preferência.

Métricas calculadas:
- **RMSE** — Root Mean Squared Error: penaliza erros grandes
- **MAE** — Mean Absolute Error: erro médio absoluto
- **MSE** — Mean Squared Error: erro quadrático médio
- **R²** — Coeficiente de determinação: variância explicada pelo modelo

---

## Design Patterns Aplicados

**Factory Pattern** (`src/models/factory.py`)

Permite criar modelos pelo nome sem conhecer a classe concreta. Novos modelos são registrados com o decorador `@register_model("nome")`.

```python
model = create_model("mlp", n_users=943, n_items=1682)
```

**Strategy Pattern** (`src/data/preprocessors.py`)

Define uma interface comum (`PreprocessingStrategy`) para diferentes transformações. Implementações: `IdEncoder`, `RatingNormalizer`.

```python
encoder = IdEncoder()
data = encoder.fit_transform(data)
```

**SOLID aplicado:**

| Princípio | Aplicação |
|---|---|
| S — Single Responsibility | Cada módulo tem uma única responsabilidade |
| O — Open/Closed | Aberto para extensão sem modificar código existente |
| L — Liskov Substitution | Qualquer modelo substitui `BaseRecommender` |
| I — Interface Segregation | Interfaces pequenas e coesas |
| D — Dependency Inversion | Código depende de abstrações, não implementações |

---

## Docker

```bash
docker compose up --build
```

Serviços:
- `mlflow` — servidor MLflow na porta 5000
- `trainer` — container de treinamento do modelo

O `Dockerfile` usa **build multi-stage**:
- `builder` — instala dependências com uv
- `runtime` — copia apenas o necessário, imagem final otimizada

---

## Estrutura do Projeto

```
Projeto-Embedding-TC02/
├── src/
│   ├── config.py               # caminhos centralizados
│   ├── settings.py             # configurações via .env (Pydantic)
│   ├── models/
│   │   ├── base.py             # interface BaseRecommender (ABC)
│   │   ├── factory.py          # Factory Pattern
│   │   ├── mlp.py              # MLP com embeddings (PyTorch)
│   │   └── trainer.py          # loop de treinamento com early stopping
│   └── data/
│       └── preprocessors.py    # Strategy Pattern: IdEncoder, RatingNormalizer
├── scripts/
│   ├── download_data.py        # baixa o dataset MovieLens 100k
│   ├── preprocess.py           # stage 1: limpeza dos dados
│   ├── feature_engineering.py  # stage 2: encoding e normalização
│   ├── train.py                # stage 3: treino + MLflow logging
│   ├── evaluate.py             # stage 4: métricas finais
│   ├── register_model.py       # registra modelo no MLflow Registry
│   └── validate_env.py         # valida instalação do ambiente
├── data/
│   ├── raw/                    # dados brutos (versionados com DVC)
│   └── processed/              # features processadas
├── models/                     # artefatos treinados (.pt)
├── metrics/                    # métricas em JSON
├── dvc.yaml                    # pipeline reprodutível
├── docker-compose.yml          # orquestração de serviços
├── Dockerfile                  # build multi-stage
├── pyproject.toml              # dependências do projeto
├── uv.lock                     # lock file
├── .env.example                # template de variáveis
└── MODEL_CARD.md               # documentação do modelo
```

---

## Qualidade de Código

```bash
# Verificar linting
uv run ruff check src/

# Corrigir automaticamente
uv run ruff check src/ --fix

# Ativar pre-commit hooks
uv run pre-commit install

# Rodar manualmente em todos os arquivos
uv run pre-commit run --all-files
```

Padrões seguidos:
- Funções com no máximo 20 linhas
- Type hints em todas as funções públicas
- Docstrings no estilo Google
- Commits semânticos (`feat`, `fix`, `chore`, `docs`)
- Naming conventions PEP8

---

## Licença

MIT License — livre para uso acadêmico e comercial.

---

## Apresentação

link de acesso[https://youtu.be/pNP_pRuBJrk]