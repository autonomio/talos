# AutoScan

`AutoScan()` provides a streamlined way for conducting a hyperparameter search experiment with any dataset. It is particularly useful for early exploration as with default settings `AutoScan()` casts a very broad parameter space including all common hyperparameters, network shapes, sizes, as well as architectures

Configure the `AutoScan()` experiment and then use the property `start` in the returned class object to start the actual experiment.

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

## AutoScan Arguments

Argument | Input | Description
--------- | ------- | -----------
`task` | str or None | `binary`, `multi_label`, `multi_class`, or `continuous`
`experiment_name` | str | Name shared with the resulting `Scan()`
`max_param_values` | int | Number of parameter values to be included

Set `task` according to the prediction problem. For custom metrics with `task=None`, construct `AutoModel(task=None, experiment_name=..., metric=[...])` and pass its model through `auto.start(model=...)`.

## AutoScan Properties

The only property **`start`** starts the actual experiment. `AutoScan.start()` accepts the following arguments:

Argument | Input | Description
--------- | ------- | -----------
`x` | array or list of arrays | prediction features
`y` | array or list of arrays | prediction outcome variable
`kwargs` | arguments | any `Scan()` argument can be passed into `AutoScan.start()`
