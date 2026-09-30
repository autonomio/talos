"""Supply context with x_train/y_train and optional x_val/y_val; edit freely."""
backend = 'keras'

import numpy as np


def params():
    return {'units': [16, 32], 'epochs': [5], 'batch_size': [32], 'learning_rate': [0.001]}


def prep(context, round_params=None):
    if context is None:
        raise ValueError('Supply caller data to execute(), or implement data loading in this prep')
    return context


def model(prepared, round_params):
    import keras
    x = np.asarray(prepared['x_train'])
    y = np.asarray(prepared['y_train'])
    task = prepared.get('task', 'regression')
    if task == 'multiclass':
        outputs = y.shape[-1] if y.ndim > 1 else len(np.unique(y))
        activation = 'softmax'
        loss = 'categorical_crossentropy' if y.ndim > 1 else 'sparse_categorical_crossentropy'
    elif task == 'binary':
        outputs, activation, loss = 1, 'sigmoid', 'binary_crossentropy'
    else:
        outputs = y.shape[-1] if y.ndim > 1 else 1
        activation, loss = 'linear', 'mse'
    network = keras.Sequential([keras.layers.Input(shape=x.shape[1:]),
                                keras.layers.Flatten(),
                                keras.layers.Dense(round_params['units'], activation='relu'),
                                keras.layers.Dense(outputs, activation=activation)])
    optimizer = keras.optimizers.Adam(learning_rate=round_params['learning_rate'])
    network.compile(optimizer=optimizer, loss=prepared.get('loss', loss),
                    metrics=prepared.get('metrics', []))
    validation = (prepared['x_val'], prepared['y_val']) if 'x_val' in prepared else None
    history = network.fit(x, y, validation_data=validation,
                          epochs=round_params['epochs'], batch_size=round_params['batch_size'], verbose=0)
    return history, network
