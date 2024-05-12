from kserve import Model, ModelServer, model_server
import argparse
from typing import Dict

from db_config import *
import mysql.connector as sql
import pandas as pd

from pymilvus import MilvusClient
import json
import numpy as np

class SampleTransformer(Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.ready = True

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        print("============== preprocess ==============")
        db_connection = sql.connect(host=HOST, database=DATABASE_NAME, user=USERNAME, password=PASSWORD)
        articles_fv = pd.read_sql('SELECT * FROM rec_articles', con=db_connection)
        articles_features = articles_fv.columns.to_list()   
        print(articles_features)
        print("======================================")
        client = MilvusClient(uri='http://milvus.dev.sinsang.market:19530')
        emb_dim = 16
        embedding = np.random.rand(emb_dim)
        res = client.search(
        collection_name="rec_candidate", 
        data=[embedding], 
        ann_fields="vector",
        limit=20,
        output_fields=["id"]
        )
        print(json.dumps(res, indent=2))
        print([item['id'] for item in res[0]])

        return inputs

    def postprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        print("============== preprocess ==============")
        return inputs

parser = argparse.ArgumentParser(parents=[model_server.parser], conflict_handler='resolve')
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function"
)
parser.add_argument(
    "--model_name", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()        

if __name__ == "__main__":
    model = SampleTransformer(args.model_name, predictor_host=args.predictor_host)
    ModelServer(workers=1).start([model])