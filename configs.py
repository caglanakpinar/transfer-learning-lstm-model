import numpy as np

debug_run = True
metrics = {
            'days': list(range(1, 8)),
            'hours': list(range(24)),
            'stores': ['store_' + str(i) for i in range(1, 3)],
            'mins': list(range(60))
}


model_features = {'ratios': {
                                'd_ratios': list(np.arange(0.8, 1, 0.05)),
                                'h_ratios': list(np.arange(0.8, 1, 0.05)),
                                's_ratios': list(np.arange(0.8, 1, 0.05)),
                                '_ratios': {'w_h_s_ratio': None}
                             },
                    'login': {
                              'd_ratios': list(np.arange(0.05, 0.75, 0.1)),
                              'h_ratios': list(np.arange(0.05, 0.75, 0.1)),
                              's_ratios': list(np.arange(0.05, 0.75, 0.1)),
                              '_ratios': {'w_h_s_ratio': None}
                             },

                    'basket': {
                               'd_ratios': list(np.arange(0.05, 0.6, 0.09)),
                               'h_ratios': list(np.arange(0.05, 0.6, 0.09)),
                               's_ratios': list(np.arange(0.05, 0.6, 0.09)),
                               '_ratios': {'w_h_s_ratio': None}
                              },
                    'payment_screen': {
                                 'd_ratios': list(np.arange(0.02, 0.2, 0.06)),
                                 'h_ratios': list(np.arange(0.02, 0.2, 0.06)),
                                 's_ratios': list(np.arange(0.02, 0.2, 0.06)),
                                 '_ratios': {'w_h_s_ratio': None}
                                             }
}
min_ranges = list(zip(list(np.arange(0, 65, 5)[1:]), list(np.arange(0, 60, 5)), list(range(12))))
pattern_ratios = list(np.arange(0.05, 1, 0.1) * -1) + list(np.arange(0.05, 1, 0.1)) + [0] * 2
weeks = list(range(24))[0:5]

_row = {'day': None, 'hour': None, 'store': None,
        'ratios': None, 'login': None,
        'basket': None, 'payment_screen': None}

### hyper parameters
parameters = {
              'tsteps': 1, 'lahead': 5, 'batch_size': 60 - 5 + 1,
              }
### hyper parameters
model_params = {
              'split_ratio': 0.8,
              'lstm_parameters': {
                  'tsteps': 1, 'lahead': 5, 'batch_size': 60,
                  'units': 30, 'l1': 0.01, 'l2': 0.02, 'loss': 0.01,
                  'activation': 'tanh'
              },
              'cat_features': {
                  'batch_size': 60, 'auto_encoder_units': [60, 30 , 60],
                  'l1': 0.01, 'l2': 0.02, 'loss': 0.01,
                  'auto_encoder_activation': ['relu', 'relu', 'sigmoid']
              },
              'output_feature': {
                  'batch_size': 60, 'activation': 'sigmoid',
              }
}

split_ratio = 0.2
lstm_features = ['basket', 'login', 'payment_screen', 'ratios']
target = ['close']
cat_features = ['hour', 'day']

### hyper parameters
parameters = {
              'tsteps': 1, 'lahead': 5, 'batch_size': 60 - 5 + 1,
              }
### hyper parameters
default_model_params = {
                        'split_ratio': 0.8,
                        'lstm_parameters': {
                            'tsteps': 1, 'lahead': 5, 'batch_size': 224,
                            'units': 30, 'l1': 0.01, 'l2': 0.02, 'loss': 0.01,
                            'activation': 'tanh'
                        },
                        'cat_features': {
                            'batch_size': 224, 'auto_encoder_units': [60, 30 , 60],
                            'l1': 0.01, 'l2': 0.02, 'loss': 0.01,
                            'auto_encoder_activation': ['relu', 'relu', 'sigmoid']
                        },
                        'output_feature': {
                            'batch_size': 224, 'activation': 'sigmoid',
                        }
}

hyper_parameters = {'neurons':[5, 20, 40],
                    'batch_size': [448, 336, 224],
                    'epochs': [5, 10, 15],
                    'activation': ['elu', 'relu', 'tanh']
                    } if not debug_run else {'neurons':[40, 20],
                                             'batch_size': [110, 220],
                                             'epochs': [5, 10],
                                             'activation': ['relu', 'tanh']
                                             }



