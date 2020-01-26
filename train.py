import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, concatenate
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
tf.keras.callbacks
import talos
import numpy as np
import datetime

from configs import metrics, lstm_features, target, hyper_parameters, default_model_params, debug_run
from data_access import data_read_write_to_json
from utils import reshape_2, reshape_3


def parameter_optimization_model_train(x_train, y_train, x_test, y_test, params):
    model = Sequential()
    model.add(LSTM(params['neurons']))
    model.add(Dense(1, activation=params['activation']))
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['mse'])
    history = model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'])
    return history, model


class Train:
    def __init__(self):
        self.model_params = []
        self.default_model_params = default_model_params
        self.locations = metrics['locations']
        self.lstm_features = lstm_features
        self.outputs = []
        self.target = target
        self.log_dir = "logs/fit/"
        self.tensorboard_callback = {}
        self.data = {}
        self.input_shape = ()
        self.input = []
        self.lstm = []
        self.merged_layers = []
        self.output_layer = []
        self.model = []
        self.optimum_paramters = {}


    def get_train_data(self, f, s):
        self.data = data_read_write_to_json("data/train_data_{}_{}.json".format(f, s), [], False)

    def model_logs(self, s):
        self.tensorboard_callback = TensorBoard(log_dir=self.log_dir + s + '_' + str(datetime.datetime.now())[0:10],
                                                histogram_freq=1)

    def model_init(self):
        self.outputs, self.inputs, self.input_dict, self.output_dict = [], [], {}, {}

    def reshape_inputs_outputs(self, f):
        self.input_dict[f] = reshape_3(np.array(self.data['x_train']))
        self.output_dict[f + '_output'] = reshape_2(np.array(self.data['y_train']))
        self.input_shape = self.input_dict[f].shape

    def lstm_features_network_build_process(self, s, f):
        self.input = Input(shape=(self.input_shape[1], 1), name=f)
        self.lstm = LSTM(self.default_model_params['lstm_parameters']['units'], #self.model_params['neurons'],
                         batch_size=self.default_model_params['lstm_parameters']['batch_size'],#self.model_params['batch_size'],
                         kernel_regularizer=keras.regularizers.l2(self.default_model_params['lstm_parameters']['l2']),
                         activity_regularizer=keras.regularizers.l1(self.default_model_params['lstm_parameters']['l1'])
                         )(self.input)
        self.lstm = Dense(1, activation=self.default_model_params['lstm_parameters']['activation'], # self.model_params['activation'],
                          name=f + '_output')(self.lstm)
        self.outputs.append(self.lstm)
        self.inputs.append(self.input)

    def lstm_features_network_parameter_tuning_process(self, f, s):
        print(list(self.data.keys()))
        if 'parameters' not in list(self.data.keys()):
            print("hyper parameter tuning is stated!! Optimum parameters are not found!!!")
            t = talos.Scan(x=np.array((self.data['x_train'])),
                           y=np.array((self.data['y_train'])),
                           model=parameter_optimization_model_train,
                           params=hyper_parameters,
                           experiment_name='lstm_model',
                           round_limit=10 if not debug_run else 2)
            self.optimum_paramters = t.data.sort_values(by='loss', ascending=True).to_dict('results')[0]

            self.data['parameters'] = self.optimum_paramters
            data_read_write_to_json("data/train_data_{}_{}.json".format(f, s), self.data, True)
        else:
            self.optimum_paramters = self.data['parameters']

    def multi_layer_perceptron_for_merged_lstm(self):
        self.merged_layers = concatenate(self.outputs)
        self.merged_layers = Dense(100, activation='tanh')(self.merged_layers)
        self.merged_layers = Dense(50, activation='tanh')(self.merged_layers)
        self.output_layer = Dense(1, activation='sigmoid', name='close')(self.merged_layers)
        self.model = Model(inputs=self.inputs, outputs=self.outputs + [self.output_layer])

    def model_compute(self):
        self.model.compile(optimizer='rmsprop',
                           loss={'close': 'binary_crossentropy', 'basket_output': 'mse',
                                 'login_output': 'mse', 'payment_screen_output': 'mse', 'ratios_output': 'mse'},
                      loss_weights={'close': 0.2}, metrics=['acc']
                      )
        # parameter tunning
        self.model.fit(self.input_dict,
                       self.output_dict,
                       validation_split=0.2,
                       epochs=20, batch_size=1000,
                       callbacks=[self.tensorboard_callback]
                       )

    def write_keras_model_to_json(self, s):
        with open("models/model_{}.json".format(s), "w") as json_file:
            json_file.write(self.model.to_json())

    def compute_locations_models(self):
        for s in self.locations:
            print(s, " model train ", "*" * 20)
            self.model_init()
            self.model_logs(s)
            for f in self.lstm_features:
                self.get_train_data(f, s)
                self.reshape_inputs_outputs(f)
                self.lstm_features_network_build_process(s, f)
            self.multi_layer_perceptron_for_merged_lstm()
            for f in self.target:
                self.get_train_data(f, s)
                self.output_dict['close'] = np.array(self.data['y_train'])
            self.model_compute()
            self.write_keras_model_to_json(s)
