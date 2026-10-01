# AutoScan

`talos.autom8.AutoScan` combines [AutoParams](AutoParams.md), [AutoModel](AutoModel.md) and [Scan](Scan.md) for early architecture exploration. Import it through `talos.autom8`. Default parameter generation covers common hyperparameters, network shapes, sizes and architectures; a small explicit parameter dictionary makes the first run bounded.

Install [TensorFlow](Backends.md) for the default callback, or choose an installed Keras backend through `start(backend=...)`. The caller owns dataset acquisition and scientific train/validation/test separation.

Constructing AutoScan stores configuration. Calling its `start()` method runs the experiment synchronously and returns the Scan object.

```python
import talos
from sklearn.datasets import load_iris
x, y = load_iris(return_X_y=True)
auto = talos.autom8.AutoScan(task='multi_class', experiment_name='iris_autoscan', max_param_values=2)
p = talos.autom8.AutoParams(task='multi_class', network=False, resample_params=1).params
p.update({'epochs': [1], 'first_neuron': [8], 'hidden_layers': [0],
          'batch_size': [16], 'dropout': [0.], 'losses': ['sparse_categorical_crossentropy'],
          'kernel_initializer': ['glorot_uniform'], 'activation': ['relu'],
          'shapes': ['brick'], 'lr': [1.]})
p['activation'] = ['relu', 'elu']
scan_object = auto.start(x.astype('float32'), y, params=p, round_limit=2, seed=17)
assert len(scan_object.data) == 2
```

NOTE: `auto.start()` accepts all `Scan()` arguments.

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `task` | required | `binary`, `multi_label`, `multi_class`, `continuous`, or `None` with a caller-supplied model. |
| `experiment_name` | required | Name used by the resulting Scan and default epoch logs. |
| `max_param_values` | `None` | Additional per-parameter resampling limit when parameters are generated automatically. |

Set `task` according to the prediction problem. For custom metrics or `task=None`, supply both an explicit parameter dictionary and a caller-owned training callback through `auto.start(params=..., model=...)`; the automatic presets require a supported task.

## Start the experiment

The signature is `AutoScan(task, experiment_name, max_param_values=None)` followed by `start(x, y, **kwargs)`. Start accepts:

| Argument | Input | Description |
| --------- | ------- | ----------- |
| `x` | array or list of arrays | prediction features |
| `y` | array or list of arrays | prediction outcome variable |
| `kwargs` | arguments | any `Scan()` argument can be passed into `AutoScan.start()` |

## Defaults and boundaries

An explicit `params` dictionary bypasses AutoParams and `max_param_values`. Otherwise AutoParams first applies its default four-values-per-parameter resampling, then `max_param_values` may reduce those lists further. An explicit `model` bypasses AutoModel. Remaining keyword arguments are forwarded to Scan; `experiment_name` comes from the AutoScan constructor.

The example completes two trials, returns `.data` with two result rows, and writes the same local artifacts as Scan. With generated presets, include `round_limit` or another search limit before starting. Native Torch training needs a supplied model callback because the default AutoModel rejects native Torch architecture generation.

Invalid presets, missing model parameters, framework errors and Scan input errors propagate. This helper chooses presets; it does not infer a task from labels or select a scientific evaluation protocol.

## Read next

Use [Scan](Scan.md) for all execution arguments, [AutoParams](AutoParams.md) to narrow the space, or [AutoPredict](AutoPredict.md) to compare fitted candidates on held-out data.
