# Multiple Inputs

This example highlights a toyish example on using Keras functional API for multi-input model. One would approach building an experiment around a more elaborate multi-input model exactly in the same manner as is highlighted below.

The single-file code example can be found [here](Examples_Multiple_Inputs_Code.md).

### Imports

```python
import talos
import wrangle

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.layers.merge import concatenate
```
NOTE: In this example we use another Autonomio package 'wrangle' for splitting the data.

### Loading Data
```python
x, y = talos.templates.datasets.iris()
x_train, y_train, x_val, y_val = wrangle.array_split(x, y, .5)
```
In the case of multi-input models, the data must be split into training and validation datasets before using it in `Scan()`. `x` is expected to be a list of numpy arrays and `y` a numpy array.

### Defining the Model
```python

def iris_multi(x_train, y_train, x_val, y_val, params):

    # the first side of the network
    first_input = Input(shape=(4,))
    first_hidden1 = Dense(params['left_neurons'], activation=params['activation'])(first_input)
    first_hidden2 = Dense(params['left_neurons'], activation=params['activation'])(first_hidden1)

    # the second side of the network
    second_input = Input(shape=(4,))
    second_hidden1 = Dense(params['right_neurons'], activation=params['activation'])(second_input)
    second_hidden2 = Dense(params['right_neurons'], activation=params['activation'])(second_hidden1)
    third_hidden2 = Dense(params['right_neurons'], activation=params['activation'])(second_hidden2)

    # merging the two networks
    merged = concatenate([first_hidden2, first_hidden2])

    # creating the output
    output = Dense(3, activation='softmax')(merged)

    # put the model together, compile and fit
    model = Model(inputs=[first_input, second_input], outputs=output)
    model.compile('adam',
                  'binary_crossentropy',
                  metrics=['acc', talos.utils.metrics.f1score])

    out = model.fit(x=x_train,
                    y=y_train,
                    validation_data=[x_val, y_val],
                    epochs=150,
                    batch_size=params['batch_size'],
                    verbose=0)

    return out, model
```

First, the input model must accept arguments exactly as in the example:

```python
def iris_multi(x_train, y_train, x_val, y_val, params):
```

Even though it is a multi-output model, data can be inputted to `model.fit()` as you would otherwise do it. The multi-input part will be handled later in `Scan()` as shown below.

```python
out = model.fit(x=x_train,
                y=y_train,
                ...)
```

The model must explicitly declare `validation_data` in `model.fit` because it is a multi-input model. Talos data splitting is not available for multi-input or multi-output models, or other cases where either `x` or `y` is more than 2d.

```python
out = model.fit(...
                validation_data=[x_val, y_val]
                ...)
```

Finally, the model must `return` the `model.fit` object as well as the model itself in the order of the of the example:

```python
return out, model
```


### Parameter Dictionary

```python
p = {'activation':['relu', 'elu'],
     'left_neurons': [10, 20, 30],
     'right_neurons': [10, 20, 30],
     'batch_size': [15, 20, 25]}
```

Note that the parameter dictionary allows either list of values, or tuples with range in the form `(min, max, step)`


### Scan()
```python
scan_object = talos.Scan(x=[x_train, x_train],
                         y=y_train,
                         x_val=[x_val, x_val],
                         y_val=y_val,
                         params=p,
                         model=iris_multi)
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. In the case of multi-output model, we also have to explicitly declare `val_x` and `val_y`.

The important thing to note here is that how `y` and `y_val` are handled:

```python
scan_object = talos.Scan(x=[x_train, x_train],
                         ...
                         x_val=[x_val, x_val],
                         ...)
```

Find the description for all `Scan()` arguments [here](Scan.md#scan-arguments).
