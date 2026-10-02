"""Run a small Iris sweep: python examples/iris.py (install Talos with TensorFlow)."""
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.layers import Dense, Dropout, Input, Normalization
from tensorflow.keras.models import Sequential

import talos

# Real bundled observations; no remote dataset is required.
x, labels = load_iris(return_X_y=True)
y = np.eye(3)[labels]
p = {'first_neuron': [8, 16], 'batch_size': [16, 32], 'epochs': [3]}


def iris_model(x_train, y_train, x_val, y_val, params):
    # Keep fitted preprocessing in the saved model, using only training observations.
    normalization = Normalization()
    normalization.adapt(x_train)
    model = Sequential([Input(shape=(x_train.shape[1],)), normalization,
                        Dense(params['first_neuron'], activation='relu'), Dropout(.2),
                        Dense(y_train.shape[1], activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=params['batch_size'],
                        epochs=params['epochs'], verbose=0, validation_data=(x_val, y_val))
    return history, model


if __name__ == '__main__':
    h = talos.Scan(x, y, params=p, experiment_name='iris-example', model=iris_model,
                   seed=17, disable_progress_bar=True)
    assert len(h.data) == 4
    print(h.data[['val_loss', 'val_accuracy', 'first_neuron', 'batch_size']])
