[BACK](Examples_Generator.md)

# Generator

```python
import talos
from talos.utils import SequenceGenerator

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout

x, y = ta.templates.datasets.iris()

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

p = {'activation':['relu', 'elu'],
     'optimizer': ['AdaDelta'],
     'losses': ['logcosh'],
     'shapes': ['brick'],
     'first_neuron': [32],
     'dropout': [.2, .3],
     'batch_size': [64, 128, 256],
     'epochs': [1]}

scan_object = talos.Scan(x=x_train,
                         y=y_train,
                         x_val=x_val,
                         y_val=y_val,
                         params=p,
                         model=mnist_model)
```
