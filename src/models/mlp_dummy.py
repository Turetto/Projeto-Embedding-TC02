import torch

from src.models.mlp import MLPRecommender

model = MLPRecommender(n_users=943, n_items=1682)

user_ids = torch.tensor([0, 1, 2])
item_ids = torch.tensor([10, 20, 30])

scores = model(user_ids, item_ids)
print(scores)  # 3 scores entre 0 e 1
