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
        print("============ Model loaded ============")
        print(self.model.get_metadata()['class_params'])
        print("======================================")
        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        input_data = payload["inputs"][0]
        ranking_features = input_data["ranking_features"]
        articles_id = input_data["articles_id"]
        
        input_df = pd.DataFrame(ranking_features)
        result = self.model.predict_proba(input_df)
        scores = result[:,1].tolist()
        predictions = { "scores": scores, "articles_id": articles_id }

        return {"predictions": predictions}

if __name__ == "__main__":
    model = CatBoostModel("custom-catboost-model")
    kserve.ModelServer().start([model])