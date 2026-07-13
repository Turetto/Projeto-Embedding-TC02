# Model Card — E-commerce Recommender

## Descrição
Modelo de recomendação de produtos baseado em embeddings de usuários e itens,
treinado com PyTorch sobre o dataset MovieLens 100k.

## Versão em Production
- **Modelo:** MLP v1 com embeddings
- **Versão MLflow:** 1
- **Stage:** Production
- **Experimento:** ecommerce-recommender

## Arquitetura
- **Tipo:** MLP com embeddings
- **Embedding dim:** 32
- **Camadas:** Linear(64) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Linear(1) → Sigmoid
- **Otimizador:** Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Early stopping:** patience=5

## Dataset
- **Fonte:** MovieLens 100k (GroupLens)
- **Tamanho:** 100.000 interações
- **Usuários:** 943
- **Itens:** 1.682
- **Split:** 72% treino / 8% validação / 20% teste

## Experimentos Realizados

| Modelo | RMSE | MAE | Resultado |
|---|---|---|---|
| Baseline (mean) | 0.2810 | 0.2355 | referência |
| MLP v1 | 0.2325 | 0.1828 | Production |
| MLP v2 (BatchNorm + emb 32) | 0.2408 | 0.1876 | descartado (overfitting) |

A MLP v2 foi testada com BatchNorm, embedding_dim=64 e rede mais profunda,
mas apresentou overfitting no dataset — val_loss subia enquanto train_loss
continuava caindo. O v1 manteve melhor equilíbrio bias-variância.

## Limitações
- Treinado com ratings explícitos — não captura comportamento implícito
- Cold start: não faz recomendações para usuários ou itens novos
- Dataset de filmes usado como proxy de e-commerce

## Vieses
- Itens populares tendem a receber mais recomendações
- Usuários com poucos ratings têm embeddings menos representativos

## Uso Adequado
Recomendado para ambientes com histórico suficiente de interações user-item.
Não deve ser usado como único critério de decisão em contextos críticos.