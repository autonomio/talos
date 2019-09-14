# Typical

This example highlights a typical and rather simple example of Talos experiment, and is a good starting point for those new to Talos. The single-file example can be found [here](Examples_Typical_Code.md).

### Imports

```python
import talos
from keras.models import Sequential
from keras.layers import Dense
```

### Loading Data
```python
x, y = talos.templates.datasets.iris()
```
`x` and `y` are expected to be either numpy arrays or lists of numpy arrays.

### Defining the Model
```python
def iris_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(32, input_dim=4, activation=params['activation']))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=params['optimizer'], loss=params['losses'])

    out = model.fit(x_train, y_train,
                     batch_size=params['batch_size'],
                     epochs=params['epochs'],
                     validation_data=[x_val, y_val],
                     verbose=0)

    return out, model
```

First, the input model must accept arguments exactly as in the example:

```python
def iris_model(x_train, y_train, x_val, y_val, params):
```

Second, the model must explicitly declare `validation_data` in `model.fit`:

```python
out = model.fit( ... validation_data=[x_val, y_val] ... )
```
Finally, the model must `return` the `model.fit` object as well as the model itself in the order of the of the example:

```python
return out, model
```


### Parameter Dictionary
```python
p = {'activation':['relu', 'elu'],
     'optimizer': ['Nadam', 'Adam'],
     'losses': ['logcosh'],
     'hidden_layers':[0, 1, 2],
     'batch_size': (20, 50, 5),
     'epochs': [10, 20]}
```

Note that the parameter dictionary allows either list of values, or tuples with range in the form `(min, max, step)`


### Scan()
```python
scan_object = talos.Scan(x, y, model=iris_model, params=p, fraction_limit=0.1)
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. Find the description for all `Scan()` arguments [here](Scan.md#scan-arguments).
