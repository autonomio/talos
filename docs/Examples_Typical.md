# Typical

This example highlights a typical and rather simple example of Talos experiment, and is a good starting point for those new to Talos. The single-file example can be found [here](Examples_Typical_Code.md).

### Imports

```python
import talos
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, concatenate
```

### Loading Data
```python
x, y = talos.templates.datasets.iris()
x_train, x_val, y_train, y_val = train_test_split(
    x.astype('float32'), y.astype('float32'), test_size=.2, random_state=17,
    stratify=y.argmax(axis=1))
```
`x` and `y` are expected to be either numpy arrays or lists of numpy arrays.

### Defining the Model
```python
def iris_model(x_train, y_train, x_val, y_val, params):
    model = Sequential([Input(shape=(4,)),
                        Dense(params['first_neuron'], activation=params['activation']),
                        Dense(3, activation='softmax')])
    model.compile(optimizer=params['optimizer'], loss=params['losses'],
                  metrics=['accuracy', talos.utils.metrics.f1score])
    out = model.fit(x_train, y_train, batch_size=params['batch_size'],
                    epochs=params['epochs'], validation_data=(x_val, y_val),
                    verbose=0)
    return out, model
```

First, the input model must accept arguments exactly as in the example:

`def iris_model(x_train, y_train, x_val, y_val, params):`

Second, the model must explicitly declare `validation_data` in `model.fit`:

`model.fit(x_train, y_train, validation_data=(x_val, y_val), ...)`
Finally, the model must `return` the `model.fit` object as well as the model itself in the order of the of the example:

`return out, model`


### Parameter Dictionary
```python
p = {'activation': ['relu', 'elu'],
     'first_neuron': [8], 'optimizer': ['adam'],
     'losses': ['categorical_crossentropy'],
     'batch_size': [16], 'epochs': [2]}
```

Note that the parameter dictionary allows either list of values, or tuples with range in the form `(min, max, number_of_values)`


### Scan()
```python
scan_object = talos.Scan(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                         model=iris_model, params=p, experiment_name='iris',
                         round_limit=2, seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. Find the description for all `Scan()` arguments [here](Scan.md#scan-arguments).
