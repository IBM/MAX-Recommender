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

import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX Recommender'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'ncf'
    assert metadata['name'] == 'MAX Recommender'
    assert metadata['description'] == 'Generate personalized recommendations'
    assert metadata['license'] == 'Apache V2'


def test_response():
    model_endpoint = 'http://localhost:5000/model/predict'

    data = {'user_id': "1",
            'num_results': 5}
    r = requests.post(url=model_endpoint, data=data)

    assert r.status_code == 200
    response = r.json()
    assert len(response['predictions']) == 5

    assert response['status'] == 'ok'

    # add sanity checks here


def test_predict_invalid_user_id():
    model_endpoint = 'http://localhost:5000/model/predict'

    data = {'user_id': 'aaa',
            'num_results': 5}
    r = requests.post(url=model_endpoint, data=data)
    assert r.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__])
