from kserve import Model, ModelServer, model_server
import argparse
from typing import Dict

import mysql.connector as sql
from db_config import *

import numpy as np
from datetime import datetime
import requests
import json

class QueryTransformer(Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.ready = True

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        inputs = inputs["instances"] if "instances" in inputs else inputs
        # extract month
        month_of_purchase = datetime.fromisoformat(inputs.pop("month_of_purchase"))
        # get customer features
        customer_features = self.query_customer_features(inputs["customer_id"])
        # enrich inputs
        inputs["age"] = customer_features[1]
        inputs["month_sin"] = self.month_to_sin(month_of_purchase)
        inputs["month_cos"] = self.month_to_cos(month_of_purchase)

        print("============== preprocess ==============")
        r = {"instances" : [inputs]}
        print(r)
        return r

    def postprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        print("============== postprocess ==============")
        print(inputs)
        ranking_output = self.call_ranking_model({ "instances": inputs["predictions"] })
        return {"predictions": ranking_output}

    ##########    

    def month_to_sin(self, t_dat):
        month = t_dat.month - 1
        C = 2*np.pi/12
        return np.sin(month*C).item()

    def month_to_cos(self, t_dat):
        month = t_dat.month - 1
        C = 2*np.pi/12
        return np.cos(month*C).item()

    def query_customer_features(self, customer_id):
      db_connection = sql.connect(host=HOST, database=DATABASE_NAME, user=USERNAME, password=PASSWORD)
      with db_connection.cursor() as cursor:
          query = f"SELECT * FROM rec_customers WHERE customer_id = '{customer_id}'"
          cursor.execute(query)
          rows = cursor.fetchall()

      customer_features = list(rows[0]) if len(rows) > 0 else None
      return customer_features
    
    # FIXME: 클러스터 내부 호출하기
    def call_ranking_model(self, inputs):
      model_name = 'custom-catboost-model'
      # host = 'knative-local-gateway.istio-system.svc.cluster.local'
      # host = 'localhost'
      host = 'host.minikube.internal'
      port=8080
      url = f"http://{host}:{port}/v1/models/{model_name}:predict"
      headers = {
        'HOST': f"{model_name}.kserve-test.example.com",
        'Content-Type': 'application/json'
      }
      response = requests.post(url, headers=headers, data=json.dumps(inputs))
      return response.json()

parser = argparse.ArgumentParser(parents=[model_server.parser], conflict_handler='resolve')
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function"
)
parser.add_argument(
    "--model_name", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()        

if __name__ == "__main__":
    model = QueryTransformer(args.model_name, predictor_host=args.predictor_host)
    ModelServer(workers=1).start([model])