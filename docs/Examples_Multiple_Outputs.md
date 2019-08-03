# Multiple Outputs

This example highlights a slightly playful example for a multi-output model where the dataset consists of a Talos hyperparameter scan results with roughly 600 hyperparameter permutations to solve the Kaggle Telco Churn challenge. The not so obvious idea here is to use deep learning to optimize the process of optimizing deep learning process i.e. use hyperparameter optimization data to optimize hyperparameter optimization.

The single-file code example can be found [here](Examples_Multiple_Outputs_Code.md).

### Imports

```python
import talos
import wrangle

from keras.layers import Input, Dense, Dropout
from keras.models import Model

```
NOTE: In this example we use another Autonomio package 'wrangle' for splitting the data.

### Loading Data
```python
x, y = talos.templates.datasets.telco_churn()

x_train, y1_train, x_val, y1_val = wrangle.array_split(x, y[0], 0.3)
x_train, y2_train, x_val, y2_val = wrangle.array_split(x, y[1], 0.3)
```
In the case of multi-output models, the data must be split into training and validation datasets before using it in `Scan()`. `x` is expected to be a numpy array, and `y` a list of numpy arrays.

### Defining the Model
```python

def telco_churn(x_train, y_train, x_val, y_val, params):

    # the second side of the network
    input_layer = Input(shape=(42,))
    hidden_layer1 = Dense(params['neurons'], activation=params['activation'])(input_layer)
    hidden_layer2 = Dense(params['neurons'], activation=params['activation'])(hidden_layer1)
    hidden_layer3 = Dense(params['neurons'], activation=params['activation'])(hidden_layer2)

    # creating the outputs
    output1 = Dense(1, activation='sigmoid', name='loss_function')(hidden_layer3)
    output2 = Dense(1,  activation='sigmoid', name='f1_metric')(hidden_layer3)

    losses = {"loss_function": "binary_crossentropy",
              "f1_metric": "binary_crossentropy"}

    loss_weights = {"loss_function": 1.0, "f1_metric": 1.0}

    # put the model together, compile and fit
    model = Model(inputs=input_layer, outputs=[output1, output2])

    model.compile('adam', loss=losses, loss_weights=loss_weights,
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
def telco_churn(x_train, y_train, x_val, y_val, params):
```

Even though it is a multi-output model, data can be inputted to `model.fit()` as you would otherwise do it. The multi-output part will be handled later in `Scan()` as shown below.

```python
  out = model.fit(x=x_train,
                  y=y_train,
                  validation_data=[x_val, y_val]
                  ...)
```

The model must explicitly declare `validation_data` in `model.fit` because it is a multi-output model. Talos data splitting is not available for multi-input or multi-output models, or other cases where either `x` or `y` is more than 2d.

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
     'neurons': [10, 20, 30],
     'batch_size': [15, 20, 25]}
```

Note that the parameter dictionary allows either list of values, or tuples with range in the form `(min, max, step)`


### Scan()
```python
scan_object = talos.Scan(x=x_train,
                         y=[y1_train, y2_train],
                         x_val=x_val,
                         y_val=[y1_val, y2_val],
                         params=p,
                         model=telco_churn)
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. In the case of multi-output model, we also have to explicitly declare `val_x` and `val_y`.

The important thing to note here is that how `y` and `y_val` are handled:

```python
scan_object = talos.Scan(...
                         y=[y1_train, y2_train],
                         ...
                         y_val=[y1_val, y2_val],
                         ...)
```

Find the description for all `Scan()` arguments [here](Scan.md#scan-arguments).
