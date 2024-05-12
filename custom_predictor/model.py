import kserve
import joblib
import pandas as pd
from typing import Dict

class CatBoostModel(kserve.Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.load()

    def load(self):
        self.model = joblib.load('model.pkl')
        self.model.eval()
        self.ready = True

    def predict(self, payload: Dict) -> Dict:
        input_data = payload["inputs"][0]
        ranking_features = input_data["ranking_features"]
        articles_id = input_data["articles_id"]
        
        input_df = pd.DataFrame(ranking_features)
        scores = self.model.predict_proba(input_df)[:,1]
        predictions = { "scores": scores, "articles_id": articles_id }

        return {"predictions": predictions}

if __name__ == "__main__":
    model = CatBoostModel("custom-catboost-model")
    kserve.ModelServer().start([model])