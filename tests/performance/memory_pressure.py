import talos
from talos.utils import SequenceGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D

p = {'activation': ['relu'],
     'optimizer': ['AdaDelta'],
     'losses': ['categorical_crossentropy'],
     'dropout': [.2],
     'batch_size': [256],
     'epochs': [1, 1, 1, 1, 1]}

x_train, y_train, x_val, y_val = talos.templates.datasets.mnist()

@profile
def talos_version():

    def mnist_model(x_train, y_train, x_val, y_val, params):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation=params['activation'], input_shape=(28, 28, 1)))
        model.add(Flatten())
        model.add(Dense(128, activation=params['activation']))
        model.add(Dropout(params['dropout']))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=params['optimizer'],
                      loss=params['losses'],
                      metrics=['acc', talos.utils.metrics.f1score])

        out = model.fit_generator(SequenceGenerator(x_train,
                                                    y_train,
                                                    batch_size=params['batch_size']),
                                                    epochs=params['epochs'],
                                                    validation_data=[x_val, y_val],
                                                    callbacks=[],
                                                    workers=4,
                                                    verbose=0)

        return out, model

    scan_object = talos.Scan(x=x_train,
                             y=y_train,
                             x_val=x_val,
                             y_val=y_val,
                             params=p,
                             model=mnist_model,
                             experiment_name='mnist',
			                 save_weights=False)


if __name__ == "__main__":

    talos_version()
