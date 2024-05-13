# Copyright 2021 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kserve import Model, ModelServer, model_server
from pymilvus import MilvusClient
import mysql.connector as sql
import pandas as pd
from db_config import *

from typing import Dict
import argparse

class RankingTransformer(Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        # create vector db client
        self.milvus_client = MilvusClient(uri='http://milvus.dev.sinsang.market:19530')
        self.collection_name = 'rec_candidate'

        # get feature views
        db_connection = sql.connect(host=HOST, database=DATABASE_NAME, user=USERNAME, password=PASSWORD)
        with db_connection.cursor() as cursor:
            query = "SHOW COLUMNS FROM rec_articles"
            cursor.execute(query)
            columns = cursor.fetchall()
            self.articles_features = [column[0] for column in columns]

        # articles_fv = pd.read_sql('SELECT * FROM rec_articles', con=self.db_connection)
        # self.articles_features = articles_fv.columns.to_list()
        # self.customer_fv = pd.read_sql('SELECT * FROM rec_customers', con=self.db_connection)
        
        # TODO: model input schema 가져오는 부분 개선하기
        self.ranking_model_feature_names = ['age', 'month_sin', 'month_cos', 'product_type_name', 'product_group_name', 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name', 'index_name', 'index_group_name', 'section_name', 'garment_group_name']
        self.predictor_host = predictor_host
        self.ready = True

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        inputs = inputs["instances"][0]
        customer_id = inputs["customer_id"]
        
        # search for candidates
        hits = self.search_candidates(inputs["query_emb"], k=100)
        
        # get already bought items
        already_bought_items_ids = self.get_already_bought_items_ids(customer_id)

        # build dataframes
        item_id_list = []
        item_emb_list = []
        exclude_set = set(already_bought_items_ids)
        for el in hits:
            item_id = str(el["id"])
            if item_id in exclude_set:
                continue
            item_emb = el["vector"]
            item_id_list.append(item_id)
            item_emb_list.append(item_emb)
        item_id_df = pd.DataFrame({"article_id" : item_id_list})
        item_emb_df = pd.DataFrame(item_emb_list).add_prefix("item_emb_")

        # get articles feature vectors
        articles_data = []
        for article_id in item_id_list:
            article_features = self.query_article_features(article_id)
            if article_features:
                articles_data.append(article_features)
        articles_df = pd.DataFrame(data=articles_data, columns=self.articles_features)

        # join candidates with item features
        ranking_model_inputs = item_id_df.merge(articles_df, on="article_id", how="inner")
        
        # add customer features
        customer_features = self.query_customer_features(customer_id)
        ranking_model_inputs["age"] = customer_features[1]
        ranking_model_inputs["month_sin"] = inputs["month_sin"]
        ranking_model_inputs["month_cos"] = inputs["month_cos"]
        ranking_model_inputs = ranking_model_inputs[self.ranking_model_feature_names]

        return { "inputs" : [{"ranking_features": ranking_model_inputs.to_dict(orient='records'), "article_ids": item_id_list} ]}


    def postprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        print("============== preprocess ==============")
        preds = inputs["predictions"]
        ranking = list(zip(preds["scores"], preds["article_ids"])) # merge lists
        ranking.sort(reverse=True) # sort by score (descending)
        return { "ranking": ranking }

    #############

    def search_candidates(self, query_emb, k=100):
        res = self.milvus_client.search(
            collection_name=self.collection_name, 
            data=[query_emb], 
            ann_fields="vector",
            limit=k,
            output_fields=["id", "vector"]
        )
        return [item['entity'] for item in res[0]]

    def get_already_bought_items_ids(self, customer_id):
        db_connection = sql.connect(host=HOST, database=DATABASE_NAME, user=USERNAME, password=PASSWORD)
        with db_connection.cursor() as cursor:
            query = f"SELECT article_id FROM rec_transactions WHERE customer_id = '{customer_id}'"
            cursor.execute(query)
            article_ids = [item[0] for item in cursor.fetchall()]
        
        return article_ids

    # 예) ['108775015', 108775, 'Strap top', 253, 'Vest top', 'Garment Upper body', 1010016, 'Solid', 9, 'Black', 4, 'Dark', 5, 'Black', 1676, 'Jersey Basic', 'A', 'Ladieswear', 1, 'Ladieswear', 16, 'Womens Everyday Basics', 1002, 'Jersey Basic']
    def query_article_features(self, article_id):
        db_connection = sql.connect(host=HOST, database=DATABASE_NAME, user=USERNAME, password=PASSWORD)
        with db_connection.cursor() as cursor:
            query = f"SELECT * FROM rec_articles WHERE article_id = {article_id}"
            cursor.execute(query)
            rows = cursor.fetchall()

        article_features = list(rows[0]) if len(rows) > 0 else None
        return article_features

    def query_customer_features(self, customer_id):
        db_connection = sql.connect(host=HOST, database=DATABASE_NAME, user=USERNAME, password=PASSWORD)
        with db_connection.cursor() as cursor:
            query = f"SELECT * FROM rec_customers WHERE customer_id = '{customer_id}'"
            cursor.execute(query)
            rows = cursor.fetchall()

        customer_features = list(rows[0]) if len(rows) > 0 else None
        return customer_features





parser = argparse.ArgumentParser(parents=[model_server.parser], conflict_handler='resolve')
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function"
)
parser.add_argument(
    "--model_name", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()        

if __name__ == "__main__":
    model = RankingTransformer(args.model_name, predictor_host=args.predictor_host)
    ModelServer(workers=1).start([model])