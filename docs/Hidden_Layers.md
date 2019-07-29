#### Hidden Layers

Including `hidden_layers` in a model allows the use of number of hidden layers as an optimization parameter.

Each hidden layer is followed by a Dropout regularizer. If this is undesired, set dropout to 0 with ```dropout: [0]``` in the parameter dictionary.

```python
from talos.model.layers import hidden_layers

def input_model():
  # model prep and input layer...
  hidden_layers(model, params, 1)
  # rest of the model...
```

When hidden layers are used, <code>dropout, hidden_layers, and first_neuron</code> parameters must be included in the parameter dictionary.
