# AutoModel

`AutoModel` provides a meaningful way to test several network architectures in an automated manner. Currently there are five supported architectures:

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

## AutoModel Arguments

Argument | Input | Description
--------- | ------- | -----------
`task` | str or None | `binary`, `multi_label`, `multi_class`, or `continuous`
`experiment_name` | str | Name shared with `Scan()`
`backend` | str | `tensorflow` or standalone `keras`
`metric` | None or list | One or more Keras metric (functions) to be used in the model

Setting `task` affects several aspects of the model and should be set according to the specific prediction task, or set to `None` in which case `metric` input is required.

These architecture presets build Keras models. For native PyTorch training, supply a Torch callback or use the Torch SFD template.
