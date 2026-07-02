import numpy as np
from src.models.factory import register_model, create_model
from src.models.base import BaseRecommender

@register_model("dummy")
class DummyModel(BaseRecommender):
    def fit(self, user_ids, item_ids, ratings):
        print("Treinando...")
    
    def predict(self, user_id, item_ids):
        return np.zeros(len(item_ids))
    
model = create_model("dummy")
model.fit(None,None,None)

model2 = create_model("xpto")
model2.fit(None,None,None)