#!/usr/bin/env python
import talos as ta

from talos.model import lr_normalizer, early_stopper, hidden_layers

from keras.models import Sequential
from keras.layers import Dropout, Dense

from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop, Nadam
from keras.activations import softmax, relu, elu, sigmoid
from keras.losses import categorical_crossentropy, logcosh, binary_crossentropy
from talos.metrics.keras_metrics import fmeasure


def iris_model(x_train, y_train, x_val, y_val, params):

    # note how instead of passing the value, we pass a dictionary entry
    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation='relu'))

    # same here, just passing a dictionary entry
    model.add(Dropout(params['dropout']))

    # with this call we can create any number of hidden layers
    hidden_layers(model, params, y_train.shape[1])

    # again, instead of the activation name, we have a dictionary entry
    model.add(Dense(y_train.shape[1],
                    activation=params['last_activation']))

    # here are using a learning rate boundary
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],
                  params['optimizer'])),
                  loss=params['losses'],
                  metrics=['acc'])

    # here we are also using the early_stopper function for a callback
    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val],
                    callbacks=early_stopper(params['epochs'], mode='strict'))

    return out, model


def cervix_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation='relu'))

    model.add(Dropout(params['dropout']))

    hidden_layers(model, params, 1)

    model.add(Dense(1, activation=params['last_activation']))

    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'],
                  metrics=['acc', fmeasure])

    results = model.fit(x_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0,
                        validation_data=[x_val, y_val],
                        callbacks=early_stopper(params['epochs'], mode='moderate', monitor='val_fmeasure'))

    return results, model



# PROGRAM STARTS HERE
# ===================

# here use a standard 2d dictionary for inputting the param boundaries
p = {'lr': (2, 10, 30),
     'first_neuron': [4, 8, 16, 32, 64, 128],
     'hidden_layers': [2, 3, 4, 5, 6],
     'batch_size': [2, 3, 4],
     'epochs': [1],
     'dropout': (0, 0.40, 10),
     'optimizer': [Adam, Nadam, SGD, Adadelta, Adagrad, RMSprop, Nadam, Adamax],
     'losses': [categorical_crossentropy, logcosh],
     'activation': [relu, elu],
     'last_activation': [softmax],
     'weight_regulizer': [None],
     'emb_output_dims': [None]}

x, y = ta.datasets.iris()

h = ta.Scan(x, y,
            params=p,
            dataset_name='testing',
            experiment_no='000',
            model=iris_model,
            grid_downsample=.0001,
            reduction_method='spear',
            reduction_interval=5)

r = ta.Reporting('testing_000.csv')

# here use a standard 2d dictionary for inputting the param boundaries

x, y = ta.datasets.cervical_cancer()
p = {'lr': (0.5, 5, 10),
     'first_neuron': [4, 8, 16, 32, 64],
     'hidden_layers': [2,3,4,5],
     'batch_size': (2, 30, 10),
     'epochs': [3],
     'dropout': (0, 0.5, 5),
     'weight_regulizer': [None],
     'emb_output_dims': [None],
     'optimizer': [Nadam],
     'loss': [logcosh, binary_crossentropy],
     'activation': [relu],
     'last_activation': [sigmoid]}

ta.Scan(x, y,
        grid_downsample=0.001,
        params=p,
        dataset_name='cervix',
        experiment_no='a',
        model=cervix_model, reduction_method='spear', reduction_interval=5)

ta.Reporting('cervix_a.csv')
