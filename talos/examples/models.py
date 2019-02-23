#!/usr/bin/env python


def breast_cancer(x_train, y_train, x_val, y_val, params):

    from keras.models import Sequential
    from keras.layers import Dropout, Dense
    from talos.model import lr_normalizer, early_stopper, hidden_layers

    from talos.metrics.keras_metrics import matthews_correlation_acc, precision_acc
    from talos.metrics.keras_metrics import recall_acc, fmeasure_acc

    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation='relu'))

    model.add(Dropout(params['dropout']))

    hidden_layers(model, params, 1)

    model.add(Dense(1, activation=params['last_activation']))

    model.compile(optimizer=params['optimizer']
                  (lr=lr_normalizer(params['lr'],
                                    params['optimizer'])),
                  loss=params['losses'],
                  metrics=['acc',
                           fmeasure_acc,
                           recall_acc,
                           precision_acc,
                           matthews_correlation_acc])

    results = model.fit(x_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0,
                        validation_data=[x_val, y_val],
                        callbacks=[early_stopper(params['epochs'],
                                                 mode='moderate',
                                                 monitor='val_fmeasure')])

    return results, model


def cervical_cancer(x_train, y_train, x_val, y_val, params):

    from keras.models import Sequential
    from keras.layers import Dropout, Dense
    from talos.model import lr_normalizer, early_stopper, hidden_layers

    from talos.metrics.keras_metrics import matthews_correlation_acc, precision_acc
    from talos.metrics.keras_metrics import recall_acc, fmeasure_acc

    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation='relu'))

    model.add(Dropout(params['dropout']))

    hidden_layers(model, params, 1)

    model.add(Dense(1, activation=params['last_activation']))

    model.compile(optimizer=params['optimizer']
                  (lr=lr_normalizer(params['lr'],
                                    params['optimizer'])),
                  loss=params['losses'],
                  metrics=['acc',
                           fmeasure_acc,
                           recall_acc,
                           precision_acc,
                           matthews_correlation_acc])

    results = model.fit(x_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0,
                        validation_data=[x_val, y_val],
                        callbacks=[early_stopper(params['epochs'],
                                                 mode='moderate',
                                                 monitor='val_fmeasure')])

    return results, model


def titanic(x_train, y_train, x_val, y_val, params):

    from keras.models import Sequential
    from keras.layers import Dropout, Dense

    # note how instead of passing the value, we pass a dictionary entry
    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation='relu'))

    # same here, just passing a dictionary entry
    model.add(Dropout(params['dropout']))

    # again, instead of the activation name, we have a dictionary entry
    model.add(Dense(1, activation=params['last_activation']))

    # here are using a learning rate boundary
    model.compile(optimizer=params['optimizer'],
                  loss=params['losses'],
                  metrics=['acc'])

    # here we are also using the early_stopper function for a callback
    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=2,
                    verbose=0,
                    validation_data=[x_val, y_val])

    return out, model


def iris(x_train, y_train, x_val, y_val, params):

    from keras.models import Sequential
    from keras.layers import Dropout, Dense
    from talos.model import lr_normalizer, early_stopper, hidden_layers

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
    model.compile(optimizer=params['optimizer']
                  (lr=lr_normalizer(params['lr'],
                                    params['optimizer'])),
                  loss=params['losses'],
                  metrics=['acc'])

    # here we are also using the early_stopper function for a callback
    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val],
                    callbacks=[early_stopper(params['epochs'], mode=[1, 1])])

    return out, model
