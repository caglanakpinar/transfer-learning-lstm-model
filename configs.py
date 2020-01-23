import numpy as np

metrics = {
            'days': list(range(1, 8)),
            'hours': list(range(24)),
            'stores': ['store_' + str(i) for i in range(1, 51)],
            'mins': list(range(60))
}


model_features = {'ratios': {
                                'd_ratios': list(np.arange(0.75, 1, 0.005)),
                                'h_ratios': list(np.arange(0.75, 1, 0.005)),
                                's_ratios': list(np.arange(0.75, 1, 0.005)),
                                '_ratios': {'w_h_s_ratio': None}
                             },
                    'login': {
                              'd_ratios': list(np.arange(0.05, 0.75, 0.005)),
                              'h_ratios': list(np.arange(0.05, 0.75, 0.005)),
                              's_ratios': list(np.arange(0.05, 0.75, 0.005)),
                              '_ratios': {'w_h_s_ratio': None}
                             },

                    'basket': {
                               'd_ratios': list(np.arange(0.05, 0.4, 0.005)),
                               'h_ratios': list(np.arange(0.05, 0.4, 0.005)),
                               's_ratios': list(np.arange(0.05, 0.4, 0.005)),
                               '_ratios': {'w_h_s_ratio': None}
                              },
                    'payment_screen': {
                                 'd_ratios': list(np.arange(0.02, 0.2, 0.005)),
                                 'h_ratios': list(np.arange(0.02, 0.2, 0.005)),
                                 's_ratios': list(np.arange(0.02, 0.2, 0.005)),
                                 '_ratios': {'w_h_s_ratio': None}
                                             }
}
min_ranges = list(zip(list(np.arange(0, 65, 5)[1:]), list(np.arange(0, 60, 5)), list(range(12))))
pattern_ratios = list(np.arange(0.05, 1, 0.05) * -1) + list(np.arange(0.05, 1, 0.05)) + [0] * 5
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
output = ['close']
cat_features = ['hour', 'day']