import torch
import torch.nn as nn

from src.models.base import BaseRecommender
from src.models.factory import register_model


@register_model("mlp")
class MLPRecommender(BaseRecommender, nn.Module):
    """
    Modelo de rede neural com embeddings para recomendacao
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32) -> None:
        nn.Module.__init__(self)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da rede
        """

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.network(x).squeeze()

    def fit(self, user_ids, item_ids, ratings) -> None:
        pass

    def predict(self, user_id, item_ids) -> None:
        pass

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
