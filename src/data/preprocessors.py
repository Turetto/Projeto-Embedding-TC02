from abc import ABC, abstractmethod

import pandas as pd

class PreProcessingStrategy(ABC):
    """
    Interface base para estratégias de pré processamento
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "PreProcessingStrategy":
        """
        Treinamento dos parâmentos a partir de um banco de dados
        """
        ...

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        transformação de dados
        """
        ...
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Treinamento e transformação de dados simultaneamente
        """
        return self.fit(data).transform(data)
    
class IdEncoder(PreProcessingStrategy):
    """
    Converter user_id e item_id para indices inteiros continuos
    """

    def __init__(self) -> None:
        self._user_map: dict = {}
        self._item_map: dict = {}

    def fit(self, data: pd.DataFrame) -> "IdEncoder":
        users = sorted(data["user_id"].unique())
        items = sorted(data["item_id"].unique())
        self._user_map = {uid: idx for idx, uid in enumerate(users)}
        self._item_map = {iid: idx for idx, iid in enumerate(items)}
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["user_id"] = data["user_id"].map(self._user_map)
        result["item_id"] = data["item_id"].map(self._item_map)
        return result
    
    @property
    def n_users(self) -> int:
        return len(self._user_map)
    
    @property
    def n_items(self) -> int:
        return len(self._item_map)

    