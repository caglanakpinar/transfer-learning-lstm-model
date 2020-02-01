import sys
import datetime

from data_access import get_data
from data_preparation import DataPreparation
from dashboard import create_dashboard
from prediction import Prediction
from train import Train

def main(argv):
    if argv[1] == 'train_process':
        get_data()
        data_preparation = DataPreparation()
        data_preparation.generate_data_for_model()
        train_model = Train()
        train_model.compute_locations_models()
        prediction = Prediction()
        prediction.get_models()
        create_dashboard(prediction)

if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv)