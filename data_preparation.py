import numpy as np
import pandas as pd
import os
from os import listdir

from utils import get_files
from configs import metrics, parameters, model_params, lstm_features, cat_features, target, split_ratio
from data_access import data_read_write_to_json


def compute_one_hot_encoding(df, cat_features, is_droping):
    """
    :params:
    df: data frame which cat_fatures of one-hot-encoded version will be addded as new column,
    cat_features: cat featurs whivch df has and these features are labelled or descrete features.
    :return: return df again with cat_features of one hot encoded version, total encoded features
    """
    features = []
    for col in cat_features:
        _df = pd.get_dummies(df[col], prefix=col)
        features += list(_df.columns)
        df = pd.concat([df, _df], axis=1)
        if is_droping:
            df = df.drop(col, axis=1)
    return df, features


class DataPreparation:
    def __init__(self):
        self.files_prepared_data = listdir(os.path.abspath("data/"))
        self.metrics = metrics
        self.ratio_metrics = ['days', 'hours', 'locations']
        self.parameters = parameters
        self.model_params = model_params
        self.lstm_features = lstm_features
        self.cat_features = cat_features
        self.output = target
        self.file_name = None
        self.feature_dict = {}
        self.data = []
        self._data = []
        self.weeks = None
        self.days = None
        self._x_train = None
        self._y_train = None
        self.split_ratio = split_ratio

    def create_parameters(self):
        for f in self.lstm_features + self.cat_features + self.output:
            self.feature_dict[f] = {}

    def get_locations_data(self, s):
        self.data = pd.read_csv("data/availability_ratios_{}.csv".format(s))
        self.weeks = list(self.data['week'].unique())
        self.days = list(self.data['day'].unique())
        self.hours = list(self.data['hour'].unique())
        self._x_train = None
        self._y_train = None

    def get_train_weeks(self):
            self.weeks = self.weeks[:- int(len(self.weeks) * split_ratio)]

    def query_data(self, w, d, h, f):
        self._data = self.data.query("week == @w and day == @d and hour == @h")[[f]]

    def data_preparation(self, f):
        self._y = self._data[[f]].rolling(window=self.parameters['tsteps'], center=False).mean()
        if self.parameters['lahead'] > 1:
            self._x = pd.DataFrame(np.repeat(self._data[[f]].values, repeats=self.parameters['lahead'], axis=1))
            for i, c in enumerate(self._x.columns):
                self._x[c] = self._x[c].shift(i)
        self._y = self._y[max(self.parameters['tsteps'] - 1, self.parameters['lahead'] - 1):]
        self._x = self._x[max(self.parameters['tsteps'] - 1, self.parameters['lahead'] - 1):]
        self._x_train = self._x if self._x_train is None else np.concatenate([self._x_train, self._x], axis=0)
        self._y_train = self._y if self._y_train is None else np.concatenate([self._y_train, self._y], axis=0)

    def compute_one_hot_encoding(self, f):
        self._x_train = compute_one_hot_encoding(pd.DataFrame(self._x_train).rename(columns={0: f}), f, True).values

    def generate_data_for_model(self):
        for f in self.lstm_features + self.output:
            print("feature :", f, "*" * 20)
            for s in self.metrics[self.ratio_metrics[2]]:
                print("store :", s)
                self.file_name = "train_data_{}_{}.json".format(f, s)
                if self.file_name not in self.files_prepared_data:
                    self.get_locations_data(s)
                    self.get_train_weeks()
                    for w in self.weeks:
                        for d in self.days:
                            for h in self.hours:
                                self.query_data(w, d, h, f)
                                self.data_preparation(f)

                    if f in self.lstm_features:
                        data_read_write_to_json("data/"+self.file_name,
                                                {'x_train': self._x_train.reshape((self._x_train.shape[0],
                                                                                   self._x_train.shape[1], 1)).tolist(),
                                                 'y_train': self._y_train.reshape((self._x_train.shape[0], 1)).tolist()
                                                 },
                                                True)
                    if f in self.cat_features:
                        self.compute_one_hot_encoding(f)
                        data_read_write_to_json("data/"+self.file_name,
                                                {'x_train': self._x_train},
                                                True)
                    if f in self.output:
                        data_read_write_to_json("data/"+self.file_name,
                                                {'y_train': pd.DataFrame(self._x_train).rename(columns={0: f})[[f]].values.tolist()},
                                                True)



