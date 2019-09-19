# Hidden Layers

Including `hidden_layers` in a model allows the use of number of hidden Dense layers as an optimization parameter.

Each hidden layer is followed by a Dropout regularizer. If this is undesired, set dropout to 0 with ```dropout: [0]``` in the parameter dictionary.

```python
from talos.utils import hidden_layers

def input_model():
  # model prep and input layer...
  hidden_layers(model, params, 1)
  # rest of the model...
```

When hidden layers are used, `dropout`, `shapes`, `hidden_layers`, and `first_neuron` parameters must be included in the parameter dictionary. For example:

```python

    p = {'activation':['relu', 'elu'],
         'optimizer': ['Nadam', 'Adam'],
         'losses': ['logcosh'],
         'shapes': ['brick'],          # <<< required
         'first_neuron': [32, 64],     # <<< required
         'hidden_layers':[0, 1, 2],    # <<< required
         'dropout': [.2, .3],          # <<< required
         'batch_size': [20,30,40],
         'epochs': [200]}

```

## hidden_layers Arguments

Parameter | type | Description
--------- | ------- | -----------
`model` | class object | a `Scan` object
`params` | dict  | The input model parameters dictionary
`output_dims` | int | Number of dimensions on the output layer

NOTE: `params` here refers to the dictionary where parameters of a single permutation are contained.

# Shapes

Talos allows several options for testing network architectures as a parameter. `shapes` is invoked by including it in the parameter dictionary:

```python

    # alternates between two preset shapes
    p = {...
         'shapes': ['brick', 'triangle', 'funnel'],
         ...}

    # a linear contraction of set value from input layer to output layer
    p = {...
         'shapes': [.1, .15, .2, .25],
         ...}

```
NOTE: You must use `hidden_layers` as per described above in order to leverage `shapes`.
