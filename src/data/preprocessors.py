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