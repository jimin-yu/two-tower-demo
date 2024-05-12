from kserve import Model, ModelServer, model_server
import argparse
from typing import Dict

class SampleTransformer(Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.ready = True

        db_connection = sql.connect(host=HOST, database=DATABASE_NAME, user=USERNAME, password=PASSWORD)
        articles_fv = pd.read_sql('SELECT * FROM rec_articles', con=db_connection)
        articles_features = articles_fv.columns.to_list()   
        print(articles_features)

    def preprocess(self, inputs: Dict, headers: Dict[str, str] = None) -> Dict:
        print("============== preprocess ==============")
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