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

from maxfw.model import MAXModelWrapper

import pickle  # nosec - B301:blacklist - using pickle for known files
import pandas as pd
import numpy as np
import logging
from flask import abort
from config import DEFAULT_MODEL_PATH, MODEL_META_DATA as model_meta

from core.NCF import NCF

logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = model_meta

    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))

        with open('assets/user_mapping.p', 'rb') as fp:
            self.user_to_id_mapping = pickle.load(fp)  # nosec - B301:blacklist - known file

        with open('assets/item_mapping.p', 'rb') as fp:
            self.item_to_id_mapping = pickle.load(fp)  # nosec - B301:blacklist - known file

        with open('assets/parameters.p', 'rb') as fp:
            self.parameters = pickle.load(fp)  # nosec - B301:blacklist - known file

        self.users = [user for user in self.user_to_id_mapping]
        self.items = [item for item in self.item_to_id_mapping]
        self.item_ids = np.array([self.item_to_id_mapping[item] for item in self.items])
        self.len_item_ids = len(self.item_ids)

        # Load the graph
        self.model = NCF(
            n_users=self.parameters["n_users"],
            n_items=self.parameters["n_items"],
            model_type="NeuMF",
            n_factors=self.parameters["factors"],
            layer_sizes=[16, 8, 4]
        )
        self.model.load(neumf_dir="assets", alpha=0.5)

    def _pre_process(self, inp):
        return inp

    def _post_process(self, result):
        return result

    def _predict(self, input_args):
        user = input_args['user_id']
        try:
            user_id = self.user_to_id_mapping[user]
        except KeyError:
            abort(400, "Unknown user ID.")
        raw_preds = self.model.predict(np.tile(user_id, self.len_item_ids), self.item_ids, is_list=True)
        predictions = [[user, i, p] for i, p in zip(self.items, raw_preds)]
        predictions_sorted = pd.DataFrame(predictions, columns=['user', 'item', 'prediction']) \
            .nlargest(input_args['num_results'], 'prediction') \
            .to_dict('records')
        return predictions_sorted
