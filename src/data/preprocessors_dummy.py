import pandas as pd
from src.data.preprocessors import IdEncoder

data = pd.DataFrame({
    "user_id": [10,10,99,99],
    "item_id": [5,8,5,20],
    "rating": [4,3,5,2]
})

encoder = IdEncoder()
result = encoder.fit_transform(data)
print(result)
print(f"Usuários: {encoder.n_users}, itens: {encoder.n_items}")

