import sys

import data_access
import data_preparation

parameters = {
                'data_path': 'data_set.json',
                'split_ratio' : 0.8,
                'test_split_weeks' : [23, 22, 21],
                'train_split_weeks' : list(set(list(range(1,24))) - set([23, 22, 21])),
                'lstm_features': ['basket', 'login', 'payment', 'ratios'],
                'outputs': ['close'],
                'feature_dict': {},
                'data': []
}


def main(parameters):
    parameters['data'] = data_access.get_data(parameters['data_path'])
    parameters = data_preparation.split_data_set(parameters)
    data_preparation.get_features_preparation(parameters)


if __name__ == '__main__':
    parameters = data_access.get_data(sys.args[0]) if len(sys.argv) != 0 else parameters
    main(parameters)