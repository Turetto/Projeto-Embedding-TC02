# Model Card — E-commerce Recommender

## Descrição
Modelo de recomendação de produtos baseado em embeddings de usuários e itens,
treinado com PyTorch sobre o dataset MovieLens 100k.

## Arquitetura
- **Tipo:** MLP com embeddings
- **Embedding dim:** 32
- **Camadas:** Linear(64) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Linear(1) → Sigmoid
- **Otimizador:** Adam (lr=0.001)
- **Early stopping:** patience=5

## Dataset
- **Fonte:** MovieLens 100k (GroupLens)
- **Tamanho:** 100.000 interações
- **Usuários:** 943
- **Itens:** 1.682
- **Split:** 72% treino / 8% validação / 20% teste

## Métricas (teste)
| Modelo   | RMSE   | MAE    |
|----------|--------|--------|
| Baseline | 0.2810 | 0.2355 |
| MLP      | 0.2366 | —      |

## Limitações
- Treinado apenas com ratings explícitos — não captura comportamento implícito (cliques, tempo de navegação)
- Cold start: não faz recomendações para usuários ou itens novos
- Dataset de filmes usado como proxy — em produção real precisaria de dados de e-commerce

## Vieses
- Itens populares tendem a receber mais recomendações
- Usuários com poucos ratings têm embeddings menos representativos

## Uso adequado
Recomendado para ambientes com histórico suficiente de interações user-item.
Não deve ser usado como único critério de decisão em contextos críticos.