import json
import os
import pandas as pd
from os import listdir

from data_generator import RandomDataGenerator


def get_data():
    files_prepared_data = listdir(os.path.abspath("data"))

    if files_prepared_data == []:
      print(" Data not found random dat generator will be started!!!")
      random_data_generator = RandomDataGenerator()
      random_data_generator.generate_random_data()


def data_read_write_to_json(file, feature_dict, writing):
    if writing:
        print("writing file - ", file)
        with open(file, "w", encoding='utf-8') as file:
            json.dump(feature_dict, file)
    else:
        print("reading file!! - ", file)
        with open(file, "r") as file:
            feature_dict = json.loads(file.read())
        return feature_dict