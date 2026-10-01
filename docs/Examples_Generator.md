# Generator

This example trains a small convolutional model on real handwritten digits from scikit-learn, using a bounded offline dataset and a Talos sequence generator.

The single-file code example can be found [here](Examples_Generator_Code.md).

### Imports

```python
import talos
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, concatenate
from talos.utils import SequenceGenerator
```
NOTE: In this example we will be using the `SequenceGenerator()` available in Talos.


### Loading Data
```python
from sklearn.datasets import load_digits
x, y = load_digits(return_X_y=True)
x = x.reshape(-1, 8, 8, 1).astype('float32') / 16
x_train, x_val, y_train, y_val = train_test_split(
    x, y, train_size=144, test_size=36, stratify=y, random_state=17)
```
`x` and `y` are expected to be either numpy arrays or lists of numpy arrays.

### Defining the Model
```python
def digits_model(x_train, y_train, x_val, y_val, params):
    model = Sequential([Input(shape=(8, 8, 1)),
                        Conv2D(4, (3, 3), activation=params['activation']),
                        Flatten(), Dense(8, activation=params['activation']),
                        Dropout(params['dropout']), Dense(10, activation='softmax')])
    model.compile(optimizer=params['optimizer'], loss=params['losses'],
                  metrics=['accuracy', talos.utils.metrics.f1score])
    batches = SequenceGenerator(x=x_train, y=y_train,
                                batch_size=params['batch_size'], backend='tensorflow')
    out = model.fit(batches, epochs=params['epochs'],
                    validation_data=(x_val, y_val), verbose=0)
    return out, model
```

First, the input model must accept arguments exactly as in the example:

`def digits_model(x_train, y_train, x_val, y_val, params):`

Second, the model must explicitly declare `validation_data` in `model.fit`:

`model.fit(batches, epochs=params["epochs"], validation_data=(x_val, y_val), ...)`

Third, the model must reference a data generator in `model.fit` exactly as it would be done in stand-alone Keras:

`batches = SequenceGenerator(x=x_train, y=y_train, batch_size=params["batch_size"])`

Use `model.fit()` with a Keras `Sequence` / `PyDataset`; modern Keras no longer accepts `workers` in `fit()`.

Finally, the model must `return` the `model.fit` object as well as the model itself in the order of the of the example:

`return out, model`


### Parameter Dictionary
```python
p = {'activation': ['relu', 'elu'], 'optimizer': ['adam'],
     'losses': ['sparse_categorical_crossentropy'], 'dropout': [.1],
     'batch_size': [16], 'epochs': [2]}
```

Note that the parameter dictionary allows either list of values, or tuples with range in the form `(min, max, number_of_values)`


### Scan()
```python
scan_object = talos.Scan(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                         model=digits_model, params=p, experiment_name='digits_generator',
                         round_limit=2, seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. In the case of `fit()` use, we also have to explicitly declare `x_val` and `y_val`.

Find the description for all `Scan()` arguments [here](Scan.md#scan-arguments).
