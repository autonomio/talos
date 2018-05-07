#!/usr/bin/env python

import hyperio as hy
import pandas as pd

from keras.utils import to_categorical
from hyperio.utils.utils import lr_normalizer, early_stopper, hidden_layers
from hyperio.data import data

from keras.models import Sequential
from keras.layers import Dropout, Dense

from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop, Nadam
from keras.activations import softmax, relu, elu
from keras.losses import categorical_crossentropy, logcosh


def iris_model(x_train, y_train, x_val, y_val, params):

    # note how instead of passing the value, we pass a dictionary entry
    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation='relu'))

    # same here, just passing a dictionary entry
    model.add(Dropout(params['dropout']))

    # with this call we can create any number of hidden layers
    hidden_layers(model, params)

    # again, instead of the activation name, we have a dictionary entry
    model.add(Dense(y_train.shape[1],
                    activation=params['last_activation']))

    # here are using a learning rate boundary
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'],
                  metrics=['acc'])

    # here we are also using the early_stopper function for a callback
    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val],
                    callbacks=early_stopper(params['epochs'], mode='strict'))

    return out

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
     'loss': [categorical_crossentropy, logcosh],
     'activation': [relu, elu],
     'last_activation': [softmax],
     'weight_regulizer': [None],
     'emb_output_dims': [None]}

x, y = data.iris()

h = hy.Hyperio(x, y,
               params=p,
               dataset_name='testing',
               experiment_no='000',
               model=iris_model,
               grid_downsample=.001,
               reduction_method='spear',
               reduction_interval=5)

r = hy.Reporting('testing_000.csv')
