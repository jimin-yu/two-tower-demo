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
import weaviate

from kserve.model import PredictorProtocol
from typing import Dict
import argparse
import numpy
from tritonclient.grpc.service_pb2 import ModelInferRequest, ModelInferResponse
from tritonclient.grpc import InferResult, InferInput


class RankingTransformer(Model):
    def __init__(self):
        super().__init__(name)
        self.weaviate_client = weaviate.Client("https://weaviate.dev.sinsang.market")
        self.candidate_index_name = 'ItemCandidate'



        self.predictor_host = predictor_host
        self.protocol = protocol
        self.ready = True

    def preprocess(self, request: Dict) -> ModelInferRequest:
        # Input follows the Tensorflow V1 HTTP API for binary values
        # https://www.tensorflow.org/tfx/serving/api_rest#encoding_binary_values
        input_tensors = [image_transform(instance) for instance in request["instances"]]

        # Transform to KServe v1/v2 inference protocol
        if self.protocol == PredictorProtocol.GRPC_V2.value:
            return self.v2_request_transform(numpy.asarray(input_tensors))
        else:
            inputs = [{"data": input_tensor.tolist()} for input_tensor in input_tensors]
            request = {"instances": inputs}
            return request

    def v2_request_transform(self, input_tensors):
        request = ModelInferRequest()
        request.model_name = self.name
        input_0 = InferInput("INPUT__0", input_tensors.shape, "FP32")
        input_0.set_data_from_numpy(input_tensors)
        request.inputs.extend([input_0._get_tensor()])
        if input_0._get_content() is not None:
            request.raw_input_contents.extend([input_0._get_content()])
        return request

    def postprocess(self, infer_response: ModelInferResponse) -> Dict:
        if self.protocol == PredictorProtocol.GRPC_V2.value:
            response = InferResult(infer_response)
            return {"predictions": response.as_numpy("OUTPUT__0").tolist()}
        else:
            return infer_response
    
    def search_candidates(self, query_emb, k=100):
        nearVector = {"vector": query_emb}
        result = self.weaviate_client.query.get(
            self.candidate_index_name, ["item_id"]
        ).with_near_vector(
            nearVector
        ).with_limit(k).do()

        


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function", required=True
)
parser.add_argument(
    "--protocol", help="The protocol for the predictor", default="v1"
)
parser.add_argument(
    "--model_name", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = ImageTransformer(args.model_name, predictor_host=args.predictor_host,
                             protocol=args.protocol)
    ModelServer(workers=1).start([model])