import numpy as np

from talos.model.layers import hidden_layers
from talos.model.normalizers import lr_normalizer

from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.layers import LSTM, Conv1D, SimpleRNN, Dense

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

        if params['network'] == 'conv1d':
            x, model = _add_conv1d(x_train, model, params['first_neuron'], x_train.shape[1])

        if params['network'] == 'lstm':
            x, model = _add_lstm(x_train, model, params['first_neuron'])

        if params['network'] == 'simplernn':
            x, model = _add_simplernn(x_train, model, params['first_neuron'])

        if params['network'] == 'dense':
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
        optimizer = params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer']))

        # compile the model
        model.compile(optimizer=optimizer,
                      loss=params['loss'],
                      metrics=['acc'])

        # fit the model
        out = model.fit(x_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0,
                        validation_data=[x_val, y_val])

        # pass the output to Talos
        return out, model


def _add_conv1d(x, model, filters, kernel_size):

    x = array_reshape_conv1d(x)
    model.add(Conv1D(filters, kernel_size))
    model.add(Flatten())

    return x, model


def _add_lstm(x, model, units):

    x = array_reshape_conv1d(x)
    model.add(LSTM(units))

    return x, model


def _add_simplernn(x, model, units):

    x = array_reshape_conv1d(x)
    model.add(SimpleRNN(units))

    return x, model
