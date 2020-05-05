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


import pickle

import pandas as pd
from pyspark.sql import SparkSession

from dataset.python_splitters import python_chrono_split
from dataset.python_evaluation import (map_at_k, ndcg_at_k,)
from dataset.spark_evaluation import SparkRankingEvaluation
from NCF import NCF


def create_dataset(data_path, checkpoint_path):

    # header = ("itemID", "userID", "rating", "timestamp")
    # df = pd.read_csv(
    #     data_path,
    #     engine="python",
    #     names=header,
    #     header=1,
    #     dtype={'itemID': str, 'userID':str, 'rating':int, 'timestamp':str}
    # )

    header = ("userID", "itemID", "rating", "timestamp")
    df = pd.read_csv(
        data_path,
        engine="python",
        names=header,
        header=None
    )

    with open(checkpoint_path + '/user_mapping.p', 'rb') as fp:
        user2id = pickle.load(fp)

    with open(checkpoint_path + '/item_mapping.p', 'rb') as fp:
        item2id = pickle.load(fp)

    # Converting items/users to IDs to store ints instead of str objects
    df['itemID'] = df['itemID'].apply(lambda item: item2id[item])
    df['userID'] = df['userID'].apply(lambda user: user2id[user])

    train, test = python_chrono_split(df, 0.80)
    return (train, test)


def load_model(data, checkpoint_path):

    with open(checkpoint_path + '/parameters.p', 'rb') as fp:
        parameters = pickle.load(fp)

    print(parameters)

    model = NCF(
        n_users=parameters["n_users"],
        n_items=parameters["n_items"],
        model_type="NeuMF",
        n_factors=parameters["factors"],
        layer_sizes=[16, 8, 4]
    )

    model.load(neumf_dir=checkpoint_path, alpha=0.5)

    with open(checkpoint_path + '/user_mapping.p', 'rb') as fp:
        model.user2id = pickle.load(fp)

    with open(checkpoint_path + '/item_mapping.p', 'rb') as fp:
        model.item2id = pickle.load(fp)

    return model


def get_predictions(model, train, test):
    # Columns not needed, dropping to save memory
    # try:
    #     train = train.drop(['timestamp'], axis=1)
    #     test = test.drop(['timestamp'], axis=1)
    # except AnalysisException:
    #     pass

    users, items, preds = [], [], []
    item = list(train.itemID.unique())
    for user in train.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list=True, is_mapped=False)))

    all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})
    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

    return all_predictions


def evaluate_model(model, train, test):

    all_predictions = get_predictions(model, train, test)
    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

    TOP_K = 10
    eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

    print("MAP:\t%f" % eval_map)
    print("NDCG:\t%f" % eval_ndcg)


def evaluate_model_spark(model, train, test):

    all_predictions = get_predictions(model, train, test)

    spark = SparkSession.builder \
        .config("spark.driver.memory", '32g') \
        .config("spark.executor.memory", '32g') \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")

    test_df = spark.createDataFrame(test)
    predictions_df = spark.createDataFrame(all_predictions)

    TOP_K = 10
    evaluations = SparkRankingEvaluation(test_df, predictions_df, k=TOP_K)
    eval_map = evaluations.map_at_k()
    eval_ndcg = evaluations.ndcg_at_k()

    print("MAP:\t%f" % eval_map)
    print("NDCG:\t%f" % eval_ndcg)
    return(eval_map, eval_ndcg)
