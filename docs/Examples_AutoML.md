# AutoML

Performing an AutoML style hyperparameter search experiment with Talos could not be any easier.

The single-file code example can be found [here](Examples_AutoML_Code.md).

### Imports

```python
import talos
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### Loading Data
```python
x, y = load_iris(return_X_y=True)
x, y = x[y < 2].astype('float32'), y[y < 2]
x, x_test, y, y_test = train_test_split(x, y, test_size=.1, stratify=y, random_state=17)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, stratify=y, random_state=17)
scaler = StandardScaler().fit(x_train)
x_train, x_val, x_test = [scaler.transform(part) for part in (x_train, x_val, x_test)]
```

`x` and `y` are expected to be either numpy arrays or lists of numpy arrays and same applies for the case where `x_train`, `y_train`, `x_val`, `y_val` is used instead.

### Defining the Model

In this case there is no need to define the model. `talos.autom8.AutoModel()` is used behind the scenes, where several model architectures fully wired for Talos are found. We simply initiate the `AutoScan()` object first:

```python
autom8 = talos.autom8.AutoScan(task='binary', experiment_name='iris_automl', max_param_values=2)
# AutoParams supplies all required keys; bound this educational run explicitly.
p = talos.autom8.AutoParams(task='binary', network=False, resample_params=1).params
p.update({'epochs': [2], 'first_neuron': [8], 'hidden_layers': [0],
          'batch_size': [16], 'dropout': [0.], 'losses': ['binary_crossentropy'],
          'activation': ['relu', 'elu']})
```

### Parameter Dictionary

The complete dictionary is generated with `AutoParams()`. The example limits epochs and network size to keep the first run small; expand these values for research.


### Scan()

The `Scan()` itself is started through the **`start`** property of the `AutoScan()` class object.

```python
scan_object = autom8.start(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                           params=p, round_limit=2, seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```
We pass data here just like we would do it in `Scan()` normally. Also, you are free to use any of the `Scan()` arguments here to configure the experiment. Find the description for all `Scan()` arguments [here](Scan.md#scan-arguments).
