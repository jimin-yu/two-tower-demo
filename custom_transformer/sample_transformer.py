from kserve import Model, ModelServer, model_server
import argparse
from typing import Dict

class SampleTransformer(Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, inputs: Dict) -> Dict:
        print("============== preprocess ==============")
        return inputs

    def postprocess(self, inputs: Dict) -> Dict:
        print("============== preprocess ==============")
        return inputs

parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function", required=True
)
parser.add_argument(
    "--model_name", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()        

if __name__ == "__main__":
    model = SampleTransformer(args.model_name, predictor_host=args.predictor_host, protocol=args.protocol)
    ModelServer(workers=1).start([model])