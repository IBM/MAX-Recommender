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

from sklearn.model_selection import ParameterGrid


class GridSearch:

    def __init__(self, model_fn, param_grid, scoring_fn):
        self.model_fn = model_fn
        self.param_grid = param_grid
        self.scoring_fn = scoring_fn

    def run(self, data):
        # TODO Verify data has a test split
        # start_time = 0
        param_defs = ParameterGrid(self.param_grid)
        results = []
        max_score = 0.0
        max_params = None

        # Create models with all definitions and save results
        for params in param_defs:
            params["n_users"] = data.n_users
            params["n_items"] = data.n_items
            model = self.model_fn(**params)
            scores = self._fit_and_score(model, data)
            results.append((params, scores))

        for result in results:
            # TODO Pass in key for metric we want to maximize
            print(result[0])
            print("\t Result: " + str(result[1]))
            if result[1][0] > max_score:
                max_score = result[1][0]
                max_params = result[0]

        print("Training complete, best performing parameters:")
        del max_params["n_users"]
        del max_params["n_items"]
        print(max_params)
        return max_params

    def _fit_and_score(self, model, data):
        model.fit(data)
        scores = self.scoring_fn(model, data.train, data.test)
        return scores
