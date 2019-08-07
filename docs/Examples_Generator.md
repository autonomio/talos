# Generator

This example highlights a typical and rather simple example of Talos experiment, and is a good starting point for those new to Talos.

The single-file code example can be found [here](Examples_Generator_Code.md).

### Imports

```python
import talos
from talos.utils import SequenceGenerator

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
```
NOTE: In this example we will be using the `SequenceGenerator()` available in Talos.


### Loading Data
```python
x_train, y_train, x_val, y_val = talos.templates.datasets.mnist()
```
`x` and `y` are expected to be either numpy arrays or lists of numpy arrays.

### Defining the Model
```python

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
```

First, the input model must accept arguments exactly as in the example:

```python
def iris_model(x_train, y_train, x_val, y_val, params):
```

Second, the model must explicitly declare `validation_data` in `model.fit`:

```python
out = model.fit_generator( ... validation_data=[x_val, y_val] ... )
```

Third, the model must reference a data generator in `model.fit_generator` exactly as it would be done in stand-alone Keras:

```python
model.fit_generator(SequenceGenerator(x_train,
                                      y_train,
                                      batch_size=params['batch_size'],
                                      ...)
```

NOTE: Set the number of `workers` in `fit_generator()` based on your system.

Finally, the model must `return` the `model.fit` object as well as the model itself in the order of the of the example:

```python
return out, model
```


### Parameter Dictionary
```python
p = {'activation':['relu', 'elu'],
     'optimizer': ['AdaDelta'],
     'losses': ['logcosh'],
     'shapes': ['brick'],
     'first_neuron': [32],
     'dropout': [.2, .3],
     'batch_size': [64, 128, 256],
     'epochs': [1]}
```

Note that the parameter dictionary allows either list of values, or tuples with range in the form `(min, max, step)`


### Scan()
```python
scan_object = talos.Scan(x=x_train,
                         y=y_train,
                         x_val=x_val,
                         y_val=y_val,
                         params=p,
                         model=mnist_model)
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. In the case of `fit_generator()` use, we also have to explicitly declare `val_x` and `val_y`.

Find the description for all `Scan()` arguments [here](Scan.md#scan-arguments).
