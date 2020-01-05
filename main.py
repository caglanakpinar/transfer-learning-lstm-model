import sys

from data_access import get_data
from data_preparation import DataPreparation


def main(argv):
    if argv == 'train_process':
        get_data()
        DataPreparation().generate_data_for_model()


if __name__ == '__main__':
    main(sys.argv)