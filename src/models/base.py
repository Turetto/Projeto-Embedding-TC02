from abc import ABC, abstractmethod

import numpy as np


class BaseRecommender (ABC):
    """
    Interface base para os modelos de recomendação
    """

    @abstractmethod
    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray):
        """
        função de treinamento dos dados fornecidos
        """
        ...

    @abstractmethod
    def predict(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """
        Projeta scores de recomendação para um usuário.
        """
        ...

