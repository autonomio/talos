# Hidden Layers

Including `hidden_layers` in a model allows the use of number of hidden Dense layers as an optimization parameter.

Each hidden layer is followed by a Dropout regularizer. If this is undesired, set dropout to 0 with ```dropout: [0]``` in the parameter dictionary.

```python
import talos
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from talos.model import hidden_layers
x, y = talos.templates.datasets.iris()

def input_model(x_train, y_train, x_val, y_val, params):
    model = Sequential([Input(shape=(4,)), Dense(params['first_neuron'], activation='relu')])
    hidden_layers(model, params, 3)
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    return history, model
```

When hidden layers are used, `dropout`, `shapes`, `hidden_layers`, and `first_neuron` parameters must be included in the parameter dictionary. For example:

```python
p = {'activation': ['relu'], 'shapes': ['brick'], 'first_neuron': [8],
     'hidden_layers': [0, 1], 'dropout': [.1], 'batch_size': [16], 'epochs': [1]}
scan_object = talos.Scan(x, y, params=p, model=input_model, experiment_name='hidden_layers',
                         round_limit=2, seed=17)
assert len(scan_object.data) == 2
```

## hidden_layers Arguments

Parameter | type | Description
--------- | ------- | -----------
`model` | Keras model | A model being built inside the callback
`params` | dict  | The input model parameters dictionary
`last_neuron` | int | Number of dimensions on the output layer

NOTE: `params` here refers to the dictionary where parameters of a single permutation are contained.

# Shapes

Talos allows several options for testing network architectures as a parameter. `shapes` is invoked by including it in the parameter dictionary:

```python
# Alternate among preset shapes or use slopes for successive layer widths.
p['shapes'] = ['brick', 'triangle', 'funnel', .1, .15, .2, .25]
p['hidden_layers'] = [2]
shape_scan = talos.Scan(x, y, params=p, model=input_model, experiment_name='hidden_shapes',
                        round_limit=2, seed=17)
assert len(shape_scan.data) == 2
```
NOTE: You must use `hidden_layers` as per described above in order to leverage `shapes`.
