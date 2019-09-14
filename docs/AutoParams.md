# AutoParams

`AutoParams()` allows automated generation of comprehensive parameter dictionary to be used as input for `Scan()` experiments as well as a streamlined way to manipulate parameter dictionaries.

#### to automatically create a params dictionary

```python
p = talos.Autom8.AutoParams().params

```
NOTE: The above example yields a very large permutation space so configure `Scan()` accordingly with `fraction_limit`.

#### an alternative way where a class object is returned

```python
param_object = talos.Autom8.AutoParams()

```

Now various properties can be accessed through `param_object`, these are detailed below. For example:

#### modifying a single parameter in the params dictionary

```python
param_object.batch_size(bottom_value=20, max_value=100, steps=10)
```

Now the modified params dictionary can be accessed through `params_object.params`

#### to append a current parameter dictionary

```python
params_dict = talos.Autom8.AutoParams(p, task='multi_label').params

```
NOTE: Note, when the dictionary is created for a prediction task other than 'binary', the `task` argument has to be declared accordingly (`binary`, `multi_label`, `multi_class`, or `continuous`).

## AutoParams Arguments

Argument | Input | Description
--------- | ------- | -----------
`params` | dict or None | If `None` then a new parameter dictionary is created
`task` | str | 'binary', 'multi_class', 'multi_label', or 'continuous'
`replace` | bool | Replace current dictionary entries with new ones.
`auto` | bool | automatically generate or append params dictionary with all available parameters.
`network` | network | If `True` several model architectures will be added
`resample_params` | int or False | The number of values per parameter

## AutoParams Properties

The **`params`** property returns the parameter dictionary which can be used as an input to `Scan()`.

The **`resample_params`** accepts `n` as input and resamples the params dictionary so that n values remain for each parameter.

All other properties relate with manipulating individual parameters in the parameter dictionary.

**`activations`** For controlling the corresponding parameter in the parameters dictionary.

**`batch_size`** For controlling the corresponding parameter in the parameters dictionary.

**`dropout`** For controlling the corresponding parameter in the parameters dictionary.

**`epochs`** For controlling the corresponding parameter in the parameters dictionary.

**`kernel_initializer`** For controlling the corresponding parameter in the parameters dictionary.

**`last_activation`** For controlling the corresponding parameter in the parameters dictionary.

**`layers`** For controlling the corresponding parameter (i.e. `hidden_layers`) in the parameters dictionary.

**`losses`** For controlling the corresponding parameter in the parameters dictionary.

**`lr`** For controlling the corresponding parameter in the parameters dictionary.

**`networks`** For controlling the Talos present network architectures (`dense`, `lstm`, `bidirectional_lstm`, `conv1d`, and `simplernn`). NOTE: the use of preset networks requires the use of the input model from `AutoModel()` for `Scan()`.

**`neurons`** For controlling the corresponding parameter (i.e. `first_neuron`) in the parameters dictionary.

**`optimizers`** For controlling the corresponding parameter in the parameters dictionary.

**`shapes`** For controlling the Talos preset network shapes (`brick`, `funnel`, and `triangle`).

**`shapes_slope`** For controlling the shape parameter with a floating point value to set the slope of the network from input layer to output layer.
