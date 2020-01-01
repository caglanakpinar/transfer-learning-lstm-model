import json
import os
import pandas as pd

from data_generator import RandomDataGenerator



def get_data(file_path):
    try:
        with open(file_path, 'r') as outfile:
            data = pd.DataFrame(json.loads(outfile.read()))
    except:
      print(" Data not found random dat generator will be started!!!")
      RandomDataGenerator.generate_random_data()
    return data