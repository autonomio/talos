from keras.models import Sequential
from keras.layers import Dropout, Dense
from ..model.normalizers import lr_normalizer
from ..model.layers import hidden_layers
from ..metrics.keras_metrics import fmeasure


def iris(x_train, y_train, x_val, y_val, params):

    '''A model that yields 100% accuracy and f1 for Iris dataset'''

    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1],
                    activation=params['activation']))
    model.add(Dropout(params['dropout']))
    hidden_layers(model, params)
    model.add(Dense(y_train.shape[1], activation=params['last_activation']))

    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['losses'],
                  metrics=['acc', fmeasure])

    out = model.fit(x_train, y_train,
                    batch_size=20,
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val])

    return out, model


# first we have to make sure to input data and params into the function
def breast_cancer(x_train, y_train, x_val, y_val, params):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    model.add(Dense(10,
                    input_dim=x_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer='normal'))
    model.add(Dropout(params['dropout']))

    # if we want to also test for number of layers and shapes, that's possible
    hidden_layers(model, params, 1)

    # then we finish again with completely standard Keras way
    model.add(Dense(1, activation=params['last_activation'], kernel_initializer='normal'))

    model.compile(loss=params['losses'],
                  # here we add a regulizer normalization function from Talos
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  metrics=['acc', fmeasure])

    history = model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0)

    # finally we have to make sure that history object and model are returned
    return history, model
