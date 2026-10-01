# Hidden layers and shapes

`talos.model.hidden_layers` appends Keras Dense and Dropout layers to a model being built inside a [Scan](Scan.md) callback. Import it from `talos.model` or `talos.utils`. It makes the number and width of hidden layers sweep parameters; it does not train the model or add the output layer.

Install a [Keras or TensorFlow backend](Backends.md). The TensorFlow example below acquires Iris through the remote [dataset template](Templates.md), then starts two bounded scans.

Each hidden layer is followed by Dropout. Set `dropout: [0]` in the candidate dictionary to disable dropping activations while retaining the layer.

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

Include `activation`, `dropout`, `shapes`, `hidden_layers`, and `first_neuron` in the parameter dictionary. The callback receives one scalar value for each candidate. For example:

```python
p = {'activation': ['relu'], 'shapes': ['brick'], 'first_neuron': [8],
     'hidden_layers': [0, 1], 'dropout': [.1], 'batch_size': [16], 'epochs': [1]}
scan_object = talos.Scan(x, y, params=p, model=input_model, experiment_name='hidden_layers',
                         round_limit=2, seed=17)
assert len(scan_object.data) == 2
```

## Arguments

| Parameter | type | Description |
| --------- | ------- | ----------- |
| `model` | Keras model | A model being built inside the callback |
| `params` | dict  | The input model parameters dictionary |
| `last_neuron` | int | Number of dimensions on the output layer |

NOTE: `params` here refers to the dictionary where parameters of a single permutation are contained.

## Shapes

Talos allows several options for testing network architectures as a parameter. `shapes` is invoked by including it in the parameter dictionary:

```python
# Alternate among preset shapes or use slopes for successive layer widths.
p['shapes'] = ['brick', 'triangle', 'funnel', .1, .15, .2, .25]
p['hidden_layers'] = [2]
shape_scan = talos.Scan(x, y, params=p, model=input_model, experiment_name='hidden_shapes',
                        round_limit=2, seed=17)
assert len(shape_scan.data) == 2
```

The `shapes` candidates affect layer widths only when the callback invokes `hidden_layers`. Merely adding the key to a parameter dictionary does not change a caller-owned architecture.

## Mutation and shape rules

The signature is `hidden_layers(model, params, last_neuron)`. It adds `params['hidden_layers']` Dense/Dropout pairs to `model` in place and returns `None`. `last_neuron` guides width calculation; it does not add a final layer. A zero hidden-layer count adds no layers.

| Shape | Width rule |
| --- | --- |
| `'brick'` | Every hidden Dense layer uses `first_neuron`. |
| `'funnel'` | Width decreases by an integer step derived from first and final width. |
| `'triangle'` | Intermediate widths between first and final width are reversed, widening toward the final hidden layer for the usual first-width-greater-than-output case. |
| float slope | Repeatedly multiply width by `1 - slope`, truncate to integers, and floor at `last_neuron`. |

The helper recognizes optional Dense settings in the per-trial dictionary: `kernel_initializer` (default `'glorot_uniform'`), `bias_initializer` (default `'zeros'`), `use_bias` (default `True`), and regularizer/constraint settings (default `None`). `activation` and dropout are required even for a trial with zero hidden layers.

Missing required keys or an unsupported shape raise `TalosParamsError`. Layer counts, width values, dropout fractions and Dense settings must also satisfy framework constraints. This helper expects a Keras model with `.add()`; it does not construct native Torch layers.

## Read next

Use [AutoModel](AutoModel.md) for preset architectures, [Scan](Scan.md#parameter-dictionary) to define candidate values, or [learning-rate normalization](Learning_Rate_Normalizer.md) to compare optimizer settings.
