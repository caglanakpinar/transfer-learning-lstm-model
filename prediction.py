import os
from os import listdir
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import model_from_json

from configs import metrics, parameters, lstm_features, target
from utils import reshape_3


class Prediction:
    def __init__(self):
        self.model_paths = [file for file in listdir(os.path.abspath("models/")) if file.split(".")[1] == 'json']
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
        self.prediction = {f: None for f in lstm_features + target}

    def write_keras_model_to_json(self, path):
        return model_from_json(open("models/" + path, "r").read())

    def get_location_name_from_file_name(self, loc):
        return loc.split("_")[1] + "_" + loc.split("_")[-1].split(".")[0]

    def get_models(self):
        for m in self.model_paths:
            self.location_models[self.get_location_name_from_file_name(m)] = {}
            self.location_models[self.get_location_name_from_file_name(m)]['model'] = self.write_keras_model_to_json(m)

    def get_historic_sequantial_data(self, min, f):
         # laheads start previous hour
        self.X = self.data[f].values[min-self.parameters['lahead']:min] # if min = 10, lahead = 5 data will be 5. min to 10
        if self.parameters['lahead'] > min:
            start = len(self.data_prev_hour[f].values) - (self.parameters['lahead']-min)
            self.X = [list(self.data_prev_hour[f].values[start:]) + list(self.data[f].values[:min])]
        else:
            self.X = [list(self.data[f].values[min-self.parameters['lahead']:min])]

    def get_dates(self):
        self.data = self.data.reset_index(drop=True).reset_index()
        self.data['date'] = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
        self.data['date'] = self.data.apply(lambda row: row['date'] + datetime.timedelta(minutes=row['index']), axis=1)

    def get_prediction_values(self, day, hour, location, week, min):
        query_str = " week == @week and day == @day and hour == @hour"
        self.data = pd.read_csv("data/availability_ratios_" + location + ".csv").query(query_str)
        self.get_dates()
        if self.parameters['lahead'] > min:
            hour = hour - 1 if hour != 0 else 23
            query_str = " week == @week and day == @day and hour == @hour"
            self.data_prev_hour = pd.read_csv("data/availability_ratios_" + location + ".csv").query(query_str)
        for f in self.lstm_features:
            self.get_historic_sequantial_data(min, f)
            self.input_dict[f] = reshape_3(np.array(self.X)) # np.array(self.X).reshape(np.array(self.X).shape[0], np.array(self.X).shape[1], 1)

        count = 0
        predictions = self.location_models[location]['model'].predict(self.input_dict)
        for f in self.lstm_features:
            self.prediction[f] = predictions[count]
            count += 1











