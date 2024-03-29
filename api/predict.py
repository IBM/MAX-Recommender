#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from core.model import ModelWrapper
from maxfw.core import MAX_API, PredictAPI
from flask_restx import fields, inputs

# Set up parser for input data (http://flask-restx.readthedocs.io/en/stable/parsing.html)
input_parser = MAX_API.parser()
input_parser.add_argument('user_id', type=str, required=True, help='User ID to generate recommendations for')
input_parser.add_argument('num_results', type=inputs.positive, required=False, default=5, help='Number of items to return')


# Creating a JSON response model: https://flask-restx.readthedocs.io/en/stable/marshalling.html#the-api-model-factory
item_prediction = MAX_API.model('ItemPrediction', {
    'user': fields.String(required=True, description='User ID'),
    'item': fields.String(required=True, description='Item ID'),
    'prediction': fields.Float(required=True, description='Predicted score')
})

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'predictions': fields.List(fields.Nested(item_prediction), description='Recommended items and scores')
})


class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Make a prediction given input data"""
        result = {'status': 'error'}

        args = input_parser.parse_args()
        preds = self.model_wrapper.predict(args)

        result['predictions'] = preds
        result['status'] = 'ok'

        return result
