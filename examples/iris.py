# first import things as you would usually
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Nadam
from keras.losses import categorical_crossentropy, logcosh
from keras.activations import relu, elu, softmax

# import talos
import talos as ta

# load rthe iris dataset
x, y = ta.datasets.iris()

# then define the parameter boundaries

p = {'lr': (2, 10, 30),
     'first_neuron': [4, 8, 16, 32, 64, 128],
     'hidden_layers': [2, 3, 4, 5, 6],
     'batch_size': [2, 3, 4],
     'epochs': [300],
     'dropout': (0, 0.40, 10),
     'weight_regulizer': [None],
     'emb_output_dims': [None],
     'optimizer': ['adam', 'nadam'],
     'losses': [categorical_crossentropy, logcosh],
     'activation': [relu, elu],
     'last_activation': [softmax]}


# then define your Keras model
def iris_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(y_train.shape[1], activation=params['last_activation']))

    model.compile(optimizer=params['optimizer'],
                  loss=params['losses'],
                  metrics=['acc'])

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val])

    return out, model


# and run the scan
h = ta.Scan(x, y,
            params=p,
            dataset_name='first_test',
            experiment_no='aaa',
            model=iris_model,
            grid_downsample=0.5)
