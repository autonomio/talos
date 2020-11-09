# first import things as you would usually
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.activations import relu, softmax

# import talos
import talos

# load rthe iris dataset
x, y = talos.templates.datasets.iris()

# then define the parameter boundaries

p = {'first_neuron': [8, 16, 32],
     'batch_size': [2, 3, 4]}


# then define your Keras model
def iris_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=50,
                    verbose=0,
                    validation_data=[x_val, y_val])

    return out, model


# and run the scan
h = talos.Scan(x, y,
               params=p,
               experiment_name='talos-debug',
               model=iris_model,
               round_limit=10)
