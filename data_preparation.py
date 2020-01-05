import numpy as np
import pandas as pd
import os
from os import listdir

from utils import get_files
from configs import metrics, parameters, model_params, lstm_features, cat_features, output, split_ratio
from data_access import data_read_write_to_json

class DataPreparation:
    def __index__(self):
        self.files_prepared_data = listdir(os.path.abspath("data/"))
        self.metrics = metrics
        self.ratio_metrics = ['days', 'hours', 'stores']
        self.parameters = parameters
        self.model_params = model_params
        self.lstm_features = lstm_features
        self.cat_features = cat_features
        self.output = output
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

    def get_stores_data(self, s):
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

        def data_preparation(data_input, params):
            tsteps, lahead = params['tsteps'], params['lahead']
            expected_output = data_input.rolling(window=tsteps, center=False).mean()
            if lahead > 1:
                data_input = np.repeat(data_input.values, repeats=lahead, axis=1)
                data_input = pd.DataFrame(data_input)
                for i, c in enumerate(data_input.columns):
                    data_input[c] = data_input[c].shift(i)
            to_drop = max(tsteps - 1, lahead - 1)
            # drop the nan
            expected_output = expected_output[to_drop:]
            data_input = data_input[to_drop:]
            return data_input.values, expected_output.values

    def compute_one_hot_encoding(self, f):
        self._x_train = pd.get_dummies(pd.DataFrame(self._x_train).rename(columns={0: f})[[f]]).values
        self.y_train = pd.get_dummies(pd.DataFrame(self._y_train).rename(columns={0: f})[[f]]).values


    def generate_data_for_model(self):
        for f in self.lstm_features + self.cat_features + self.output:
            print("feature :", l, "*" * 20)
            for s in self.metrics[self.ratio_metrics[3]]:
                print("store :", s)
                self.file_name = "train_data_{}_{}.json".format(f, s)
                if self.file_name not in self.files_prepared_data:
                    self.get_stores_data(s)
                    self.get_train_weeks()
                    for w in self.weeks:
                        for d in self.days:
                            for h in self.hours:
                                self.query_data(w, d, h)
                                self.data_preparation(f)

                    if f in self.lstm_features:
                        data_read_write_to_json(self.file_name,
                                                {'x_train': self._x_train.reshape((self._x_train.shape[0],
                                                                                   self._x_train.shape[1], 1)).tolist(),
                                                 'y_train': self._x_train.reshape((self._x_train.shape[0], 1)).tolist()
                                                 },
                                                True)
                    if f in self.cat_features:
                        self.compute_one_hot_encoding(f)
                        data_read_write_to_json(self.file_name,
                                                {'x_train': self._x_train, 'y_train': self._x_train},
                                                True)
                    if f == self.output:
                        data_read_write_to_json(self.file_name,
                                                {'y_train': self._x_train.rename(columns={0: f})[[f]].values.tolist()},
                                                True)







def split_data_set(params):
    params['train_data_set'] = params['data'].sort_values(by='week', ascending=True).query("week in @train_split_weeks")
    params['test_data_set'] = params['data'].sort_values(by='week', ascending=True
                                                         ).query("week not in @train_split_weeks")
    return params


def data_preparation(data_input, params):
    tsteps, lahead = params['tsteps'], params['lahead']
    expected_output = data_input.rolling(window=tsteps, center=False).mean()
    if lahead > 1:
        data_input = np.repeat(data_input.values, repeats=lahead, axis=1)
        data_input = pd.DataFrame(data_input)
        for i, c in enumerate(data_input.columns):
            data_input[c] = data_input[c].shift(i)
    to_drop = max(tsteps - 1, lahead - 1)
    # drop the nan
    expected_output = expected_output[to_drop:]
    expected_output[0] = expected_output[0].apply(lambda x: int(x))
    data_input = data_input[to_drop:]
    return data_input, expected_output


def implement(params, feature, x_train, y_train, x_test, y_test):
    params[feature]['input_shape'] = x_train.shape
    params[feature]['inputs'] = x_train.tolist()
    params[feature]['output_shape'] = y_train.shape
    params[feature]['output'] = y_train.tolist()

    params[feature]['test_input'] = x_test.tolist()
    params[feature]['test_output'] = y_test.tolist()
    return params


def split_data(x, y, ratio, batch_size, params, feature):
    to_train = int(len(x) * ratio)
    # tweak to match with batch_size
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]

    # tweak to match with batch_size
    to_drop = x.shape[0] % batch_size
    if to_drop > 0:
        x_test = x_test[:-1 * to_drop]
        y_test = y_test[:-1 * to_drop]

    reshape_3 = lambda x: x.values.reshape((x.shape[0], x.shape[1], 1))
    x_train = reshape_3(x_train)
    x_test = reshape_3(x_test)

    reshape_2 = lambda x: x.values.reshape((x.shape[0], 1))
    y_train = reshape_2(y_train)
    y_test = reshape_2(y_test)

    params = implement(params, feature, x_train, y_train, x_test, y_test)
    return params


def get_features_preparation(params):
    for f in params['lstm_features']:
        params['feature_dict'][f] = {}
        file_name = "data_ready_file_" + f + ".json"
        if file_name not in get_files():
            print(file_name)
            _f = feature_dict
            _x_train, _y_train = data_preparation(pd.DataFrame(np.concatenate(list(train_data_set[l]))),
                                                  parameters['lstm_parameters'])
            _f = split_data(_x_train, _y_train, parameters['split_ratio'], parameters['lstm_parameters']['batch_size'],
                            _f, l)
            data_read_write_to_json(file_name, _f, True)
            del _f


    for f in params['lstm_features']:
        params['feature_dict'][f] = {}
    input_1 = params['data'].sort_values(by=params['lstm_features'], ascending=True)
    


