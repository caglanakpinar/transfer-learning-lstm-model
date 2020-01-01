import numpy as np

metrics = [
                       {'days': list(range(1,8))},
                       {'hours': list(range(24))},
                       {'stores': ['store_' + str(i) for i in range(1, 51)]},
                       {'mins': list(range(60))}
                     ]

model_features = [{'ratios': {
                                'd_ratios': list(np.arange(0.75, 1, 0.005)),
                                'h_ratios': list(np.arange(0.75, 1, 0.005)),
                                's_ratios': list(np.arange(0.75, 1, 0.005)),
                                '_ratios': {'w_h_s_ratio': {'w_h_s_ratio': None, 'min_ratio': None}}
                             }
                    },
                  {'login': {
                              'd_login_ratios': list(np.arange(0.05, 0.75, 0.005)),
                              'h_login_ratios': list(np.arange(0.05, 0.75, 0.005)),
                              's_login_ratios': list(np.arange(0.05, 0.75, 0.005)),
                              '_ratios': {'w_h_s_ratio': {'w_h_s_ratio': None, 'min_ratio': None}}
                           }
                  },

                  {'basket': {
                      'd_basket_ratios': list(np.arange(0.05, 0.4, 0.005)),
                      'h_basket_ratios': list(np.arange(0.05, 0.4, 0.005)),
                      's_basket_ratios': list(np.arange(0.05, 0.4, 0.005)),
                      '_ratios': {'w_h_s_ratio': {'w_h_s_ratio': None, 'min_ratio': None}}
                  }
                  },
    {'payment_screen': {
        'd_payment_ratios': list(np.arange(0.02, 0.2, 0.005)),
        'h_payment_ratios': list(np.arange(0.02, 0.2, 0.005)),
        's_payment_ratios': list(np.arange(0.02, 0.2, 0.005)),
        '_ratios': {'w_h_s_ratio': {'w_h_s_ratio': None, 'min_ratio': None}}
    }
    }

]
min_ranges = list(zip(list(np.arange(0, 65, 5)[1:]), list(np.arange(0, 60, 5)), list(range(12))))
pattern_ratios = list(np.arange(0.05, 1, 0.05) * -1) + list(np.arange(0.05, 1, 0.05)) + [0] * 5
weeks = list(range(24))[0:5]

_row = {'day': None, 'hour': None, 'store': None,
        'ratios': None, 'login': None,
        'basket': None, 'payment': None}

model_parameters = ['tsteps', 'lahead', 'input_shape', 'inputs', 'output_shape', 'output', 'test_input', 'test_output']