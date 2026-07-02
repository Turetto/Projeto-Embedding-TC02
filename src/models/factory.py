from src.models.base import BaseRecommender

_registry: dict[str, type[BaseRecommender]] = {}

def register_model(name: str):
    """
    Decorador para registrar um modelo no factory
    """
    def decorator(cls: type[BaseRecommender]):
        _registry[name] = cls
        return cls
    return decorator

def create_model(model_type: str, **kwargs) -> BaseRecommender:
    """
    Cria e retorma um modelo pelo nome

    Args:
        model_type: nome do modelo registrado
        kwargs: paramentros do construtor do modelo
    """

    if model_type not in _registry:
        available = list(_registry.keys())
        raise ValueError(f"Modelo '{model_type}' não encontrado. Modelos Disponíveis: {available}")
    return _registry[model_type](**kwargs)