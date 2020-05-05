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
import os
import json
import pickle

import pandas as pd
import tensorflow as tf

from NCF import NCF
from dataset.dataset import Dataset
from dataset.python_splitters import python_chrono_split
from evaluate import evaluate_model_spark
from grid_search import GridSearch

flags = tf.app.flags
flags.DEFINE_string("data", ".", "Path to data file")
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_integer("batch_size", 128, "The size of batch [128]")
flags.DEFINE_integer("factors", 8, "The number of latent factors [8]")
flags.DEFINE_float("learning_rate", 5e-3, "The learning rate [5e-3]")
flags.DEFINE_boolean(
    "hpo", False, "Enable hyperparameter optimization [False]")
flags.DEFINE_string("delimiter", ",", "")

header = ("userID", "itemID", "rating", "timestamp")


def create_dataset(data_path, split=0.0):
    df = pd.read_csv(
        data_path,
        engine="python",
        names=header,
        header=1,
        sep=flags.FLAGS.delimiter
    )

    if split == 0.0:
        return Dataset(df)
    else:
        train, test = python_chrono_split(df, split)
        return Dataset(train, test)


def train_model(data, checkpoint_path, model_type="NeuMF", n_factors=flags.FLAGS.factors, layer_sizes=[16, 8, 4],
                n_epochs=flags.FLAGS.epoch, batch_size=flags.FLAGS.batch_size, learning_rate=flags.FLAGS.learning_rate,):

    parameters = flags.FLAGS.flag_values_dict()
    parameters["n_users"] = data.n_users
    parameters["n_items"] = data.n_items

    model = NCF(
        n_users=data.n_users,
        n_items=data.n_items,
        model_type=model_type,
        n_factors=n_factors,
        layer_sizes=layer_sizes,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    model.fit(data)
    model.save(dir_name=checkpoint_path)

    # Save ID mapping
    with open(checkpoint_path + '/user_mapping.p', 'wb') as fp:
        pickle.dump(model.user2id, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(checkpoint_path + '/item_mapping.p', 'wb') as fp:
        pickle.dump(model.item2id, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Save parameters
    with open(checkpoint_path + '/parameters.p', 'wb') as fp:
        pickle.dump(parameters, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(checkpoint_path + '/parameters.json', 'w') as fp:
        json.dump(parameters, fp)

    return model


def main():

    model_path = ""
    data_path = ""

    if "RESULT_DIR" in os.environ:
        model_path = os.environ["RESULT_DIR"]
    if "DATA_DIR" in os.environ:
        data_path = os.environ["DATA_DIR"]

    checkpoint_path = os.path.join(model_path, "model", "checkpoint")
    data_path = os.path.join(data_path, flags.FLAGS.data)

    data = create_dataset(data_path, split=0.8)

    if flags.FLAGS.hpo:
        # Check if HPO flags set
        print("Running hyperparameter optimization")
        params = {"learning_rate": [1e-3, 5e-3, 1e-2],
                  "n_factors": [8, 16, 32], "n_epochs": [50, 100]}
        grid = GridSearch(model_fn=NCF, param_grid=params,
                          scoring_fn=evaluate_model_spark)
        optimized_params = grid.run(data)
        full_data = create_dataset(data_path)
        train_model(full_data, checkpoint_path, **optimized_params)

    else:
        train_model(data, checkpoint_path)


if __name__ == "__main__":
    main()
