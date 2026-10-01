# AutoModel

`talos.autom8.AutoModel` creates a five-argument training callback for `Scan`. Import it through `talos.autom8`. The helper builds and fits Keras architectures when Scan invokes `.model`; constructing `AutoModel` does not start a scan.

Install a [Keras or TensorFlow backend](Backends.md) before using it. Its default backend is `tensorflow`. Currently there are five supported architectures:

- conv1d
- lstm
- bidirectional_lstm
- simplernn
- dense

`AutoModel` creates an input model for Scan(). Optimized for being used together with `AutoParams()` and expects one or more of the above architectures to be included in params dictionary, for example:

```python
import talos
from sklearn.datasets import load_iris
x, y = load_iris(return_X_y=True)
p = talos.autom8.AutoParams(task='multi_class', network=False, resample_params=1).params
p.update({'epochs': [1], 'first_neuron': [8], 'hidden_layers': [0],
          'batch_size': [16], 'dropout': [0.], 'losses': ['sparse_categorical_crossentropy'],
          'kernel_initializer': ['glorot_uniform'], 'activation': ['relu'],
          'shapes': ['brick'], 'lr': [1.]})
p['network'] = ['dense', 'conv1d', 'lstm']
input_model = talos.autom8.AutoModel(task='multi_class', experiment_name='iris_architectures').model
scan_object = talos.Scan(x.astype('float32'), y, params=p, model=input_model,
                         experiment_name='iris_architectures', round_limit=3, seed=17)
assert len(scan_object.data) == 3
```

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `task` | required | `binary`, `multi_label`, `multi_class`, or `continuous` for runnable presets. The constructor also accepts `None` with a metric list, subject to the output-layer boundary below. |
| `experiment_name` | required | Name shared with `Scan()`, used by the epoch-log callback. |
| `metric` | `None` | Keras metric names or objects in a list when `task=None`. |
| `backend` | `'tensorflow'` | TensorFlow/tf.keras or standalone `'keras'`. |

Setting `task` controls metric selection and output-layer construction. Choose a supported prediction task for a runnable preset.

These architecture presets build Keras models. For native PyTorch training, supply a Torch callback or use the Torch SFD template.

## Callback result and boundaries

The constructor signature is `AutoModel(task, experiment_name, metric=None, backend='tensorflow')`. Its `.model` has the callback signature `model(x_train, y_train, x_val, y_val, params)` and returns `(history, fitted_model)`. The example produces three result rows and an epoch log for each trial in the run directory.

Supply one permutation's scalar values to the callback, including the architecture, width, hidden-layer shape, dropout, optimizer class, normalized learning rate, loss, batch size, epochs and output activation. [AutoParams](AutoParams.md) supplies the expected names. Non-dense presets reshape two-dimensional inputs to `(rows, features, 1)` internally; prediction inputs must match the fitted architecture's input shape.

Classification tasks use Talos F1 and accuracy metrics; `continuous` uses MAE and accuracy. When `task=None`, `metric` must be a list and the constructor also adds accuracy, but the current generated callback has no task-independent output-layer rule: fitting that preset raises `ValueError` for an unknown model task. Use a caller-owned callback for a custom task or metric contract. Native Torch backends raise `ValueError`; framework shape, loss and parameter errors propagate from the callback. This helper is for architecture exploration and does not validate the scientific suitability of those presets for a dataset.

## Read next

Bound the candidate space with [AutoParams](AutoParams.md) and [Scan](Scan.md), or use [SFD and CLI](SFD_and_CLI.md) for an explicit model implementation.
