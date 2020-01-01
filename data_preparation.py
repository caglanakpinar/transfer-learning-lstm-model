import numpy as np
import pandas as pd

from utils import get_files


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
    


