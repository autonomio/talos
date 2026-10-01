# AutoParams

`talos.autom8.AutoParams` generates a parameter dictionary for [Scan](Scan.md) and provides methods for changing individual candidate lists. Import it through `talos.autom8`; it constructs parameters without training a model.

Automatic optimizer generation imports the chosen [framework backend](Backends.md). Install TensorFlow for the default `backend='tensorflow'`, or select `backend='keras'` or `backend='torch'` with the corresponding extra. Native Torch optimizer lists still require a caller-owned Torch model; [AutoModel](AutoModel.md) builds Keras architectures.

## Create a parameter dictionary

```python
import talos
p = talos.autom8.AutoParams().params
assert 'optimizer' in p and 'network' in p
```

NOTE: The above example yields a very large permutation space so configure `Scan()` accordingly with `fraction_limit`.

### Keep the helper object

```python
param_object = talos.autom8.AutoParams()
assert isinstance(param_object.params, dict)
```

Methods on `param_object` change individual candidate lists. For example:

### Modify a parameter

```python
param_object.batch_size(min_size=20, max_size=100, steps=10)
assert param_object.params['batch_size'] == list(range(20, 100, 10))
```

Now the modified params dictionary can be accessed through `param_object.params`

### Extend an existing dictionary

```python
params_dict = talos.autom8.AutoParams(p.copy(), task='multi_label').params
assert params_dict['losses'] == ['categorical_crossentropy']
```

Declare `task` explicitly when generating presets for a prediction problem other than `'binary'` (`binary`, `multi_label`, `multi_class`, or `continuous`).

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `params` | `None` | Create a dictionary, or start from the supplied dictionary. |
| `task` | `'binary'` | `binary`, `multi_class`, `multi_label`, or `continuous`; selects preset losses and output activations. |
| `replace` | `True` | Overwrite existing keys when parameter methods add values; `False` fills only missing keys. |
| `auto` | `True` | Generate all available parameter presets. |
| `network` | `True` | Generate architecture candidates; `False` sets `network=['dense']`. |
| `resample_params` | `4` | Keep at most this many values per parameter, or `False` to retain all values. |
| `backend` | `'tensorflow'` | Framework from which automatic optimizer classes are imported. |

## Parameter methods

The **`params`** property returns the parameter dictionary which can be used as an input to `Scan()`.

The **`resample_params`** method accepts `n` and keeps at most that many candidate values for each parameter.

The remaining methods manipulate individual parameters in the dictionary.

**`activations`** For controlling the corresponding parameter in the parameters dictionary.

**`batch_size`** For controlling the corresponding parameter in the parameters dictionary.

**`dropout`** For controlling the corresponding parameter in the parameters dictionary.

**`epochs`** For controlling the corresponding parameter in the parameters dictionary.

**`kernel_initializers`** For controlling the corresponding parameter in the parameters dictionary.

**`last_activations`** For controlling the corresponding parameter in the parameters dictionary.

**`layers`** For controlling the corresponding parameter (i.e. `hidden_layers`) in the parameters dictionary.

**`losses`** For controlling the corresponding parameter in the parameters dictionary.

**`lr`** For controlling the corresponding parameter in the parameters dictionary.

**`networks`** For controlling the Talos present network architectures (`dense`, `lstm`, `bidirectional_lstm`, `conv1d`, and `simplernn`). NOTE: the use of preset networks requires the use of the input model from `AutoModel()` for `Scan()`.

**`neurons`** For controlling the corresponding parameter (i.e. `first_neuron`) in the parameters dictionary.

**`optimizers`** For controlling the corresponding parameter in the parameters dictionary.

**`shapes`** For controlling the Talos preset network shapes (`brick`, `funnel`, and `triangle`).

**`shapes_slope`** For controlling the shape parameter with a floating point value to set the slope of the network from input layer to output layer.

## Mutation and sampling

The signature is `AutoParams(params=None, task='binary', replace=True, auto=True, network=True, resample_params=4, backend='tensorflow')`. Methods modify `.params` and return `None`; `.params` is the dictionary passed to Scan. Automatic additions operate on a supplied dictionary before resampling creates a new dictionary, so pass a copy when preserving the original input matters.

`resample_params(n)` requires a positive integer and chooses up to `n` evenly spaced positions from each candidate list, including its endpoints. It does not sample trial combinations or assign probabilities. Integer range methods use Python's exclusive upper bound; float range methods use NumPy's exclusive upper bound.

| Method | Default candidate control |
| --- | --- |
| `layers(min_layers=0, max_layers=6, steps=1)` | Hidden-layer count range. |
| `dropout(min_dropout=0, max_dropout=.85, steps=.1)` | Dropout fractions rounded to two decimal places. |
| `neurons(min_neuron=8, max_neuron=None, steps=None)` | Preset powers of two, or an integer range when maximum and step are supplied together. |
| `batch_size(min_size=8, max_size=None, steps=None)` | Preset batch sizes, or a supplied integer range. |
| `epochs(min_epochs=50, max_epochs=None, steps=None)` | Preset epoch counts, or a supplied integer range. |
| `shapes_slope(min_slope=0, max_slope=.6, steps=.1)` | Fractional contraction slopes. |
| `shapes`, `optimizers`, `activations`, `losses`, `kernel_initializers`, `lr`, `networks`, `last_activations` | Pass a candidate list, or keep the method's `'auto'` preset. |

Unsupported task names fail preset lookup. Empty or malformed candidate lists are not repaired by this helper. Even four values per parameter yield a large Cartesian space: use Scan's limits or a deliberately small dictionary before training.

## Read next

Use [AutoModel](AutoModel.md) for the matching architecture callback, [AutoScan](AutoScan.md) to combine the presets, or [optimization strategies](Optimization_Strategies.md) to choose search limits.
