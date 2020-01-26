import sys

from data_access import get_data
from data_preparation import DataPreparation
from train import Train


def main(argv):
    if argv[1] == 'train_process':
        get_data()
        data_preparation = DataPreparation()
        data_preparation.generate_data_for_model()
        train_model = Train()
        train_model.compute_stores_models()

if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv)