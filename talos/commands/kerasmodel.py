import numpy as np

from talos.model.layers import hidden_layers
from talos.model.normalizers import lr_normalizer

from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.layers import LSTM, Conv1D, SimpleRNN, Dense, Bidirectional

try:
    from wrangle.reshape_to_conv1d import reshape_to_conv1d as array_reshape_conv1d
except ImportError:
    from wrangle import array_reshape_conv1d


class KerasModel:

    def __init__(self):

        '''An input model for Scan(). Optimized for being used together with
        Params(). For example:

        Scan(x=x, y=y, params=Params().params, model=KerasModel().model)

        NOTE: the grid from Params() is very large, so grid_downsample or
        round_limit accordingly in Scan().

        '''

        self.model = self._create_input_model

    def _create_input_model(self, x_train, y_train, x_val, y_val, params):

        model = Sequential()

        if params['network'] != 'dense':
            x_train = array_reshape_conv1d(x_train)
            x_val = array_reshape_conv1d(x_val)

        if params['network'] == 'conv1d':
            model.add(Conv1D(params['first_neuron'], x_train.shape[1]))
            model.add(Flatten())

        elif params['network'] == 'lstm':
            model.add(LSTM(params['first_neuron']))

        if params['network'] == 'bidirectional_lstm':
            model.add(Bidirectional(LSTM(params['first_neuron'])))

        elif params['network'] == 'simplernn':
            model.add(SimpleRNN(params['first_neuron']))

        elif params['network'] == 'dense':
            model.add(Dense(params['first_neuron'],
                            input_dim=x_train.shape[1],
                            activation='relu'))

        model.add(Dropout(params['dropout']))

        # add hidden layers to the model
        hidden_layers(model, params, 1)

        # output layer (this is scetchy)
        try:
            last_neuron = y_train.shape[1]
        except IndexError:
            if len(np.unique(y_train)) == 2:
                last_neuron = 1
            else:
                last_neuron = len(np.unique(y_train))

        model.add(Dense(last_neuron,
                        activation=params['last_activation']))

        # bundle the optimizer with learning rate changes
        optimizer = params['optimizer'](lr=lr_normalizer(params['lr'],
                                                         params['optimizer']))

        # compile the model
        model.compile(optimizer=optimizer,
                      loss=params['losses'],
                      metrics=['acc'])

        # fit the model
        out = model.fit(x_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0,
                        validation_data=[x_val, y_val])

        # pass the output to Talos
        return out, model
