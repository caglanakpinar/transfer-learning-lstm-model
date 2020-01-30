import json
import os
from os import listdir
import pandas as pd
import numpy as np

from configs import metrics, parameters, lstm_features
from utils import reshape_3


class Prediction:
    def __init__(self):
        self.model_paths = listdir(os.path.abspath("models/"))
        self.location_models = {}
        self._model = None
        self.data_sets = None
        self.metrics = ['day', 'hour', 'location']
        self.data = pd.DataFrame()
        self.data_prev_hour = pd.DataFrame()
        self.parameters = parameters
        self.lstm_features = lstm_features
        self.X = []
        self.input_dict = {}

    def write_keras_model_to_json(self, path):
        with open("models/", + path, "r") as file:
            self._model = json.loads(file.read())

    def get_models(self):
        for m in self.model_paths:
            self.location_models[m.split("_")[1]] = {}
            self.location_models[m.split("_")[1]]['model'] = self.write_keras_model_to_json(self, m)

    def get_historic_sequantial_data(self, min, f):
        if self.parameters['lahead'] < min: # laheads start previous hour
            self.X = self.data[f].values[min-self.parameters['lahead']:min] # if min = 10, lahead = 5 data will be 5. min to 10
        else:
            start = len(self.data_prev_hour[f].values) - (self.parameters['lahead']-min)
            self.X = np.concatenate([self.X, self.data_prev_hour[f].values[start:]], axis=0)

    def get_prediction_values(self, day, hour, location, week, min):
        query_str = " week == @week and day == @day and hour == @hour"
        self.data = pd.read_csv("data/availability_ratios_" + location + ".csv").query(query_str)
        if self.parameters['lahead'] < min:
            hour = hour - 1 if hour != 0 else 23
            query_str = " week == @week and day == @day and hour == @hour"
            self.data_prev_hour = pd.read_csv("data/availability_ratios_" + location + ".csv").query(query_str)
        for f in self.lstm_features:
            self.get_historic_sequantial_data(min, f)
            self.input_dict[f] = reshape_3(np.array(self.X))
        self.location_models.predict(self.input_dict)
        print()










