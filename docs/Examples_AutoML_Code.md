# Bounded AutoML sweep: complete code

Run a two-trial binary Iris search using built-in Talos models. This page is the standalone companion to the [walkthrough](Examples_AutoML.md).

## Prerequisites and execution

Use Python 3.11–3.13 with the [TensorFlow extra](Backends.md) (`talos[tensorflow]`). The dataset is an offline scikit-learn fixture. From a writable experiment directory, save the following program as `automl_example.py` and execute `python automl_example.py`.

The walkthrough owns the data split, callback explanation and interpretation of metrics. The program is bounded to two trials; successful execution passes its result-row assertion.

## Program

```python
import talos
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x, y = load_iris(return_X_y=True)
x, y = x[y < 2].astype('float32'), y[y < 2]
x, x_test, y, y_test = train_test_split(x, y, test_size=.1, stratify=y, random_state=17)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, stratify=y, random_state=17)
scaler = StandardScaler().fit(x_train)
x_train, x_val, x_test = [scaler.transform(part) for part in (x_train, x_val, x_test)]

autom8 = talos.autom8.AutoScan(task='binary', experiment_name='iris_automl', max_param_values=2)
# AutoParams supplies all required keys; bound this educational run explicitly.
p = talos.autom8.AutoParams(task='binary', network=False, resample_params=1).params
p.update({'epochs': [2], 'first_neuron': [8], 'hidden_layers': [0],
          'batch_size': [16], 'dropout': [0.], 'losses': ['binary_crossentropy'],
          'activation': ['relu', 'elu']})

scan_object = autom8.start(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                           params=p, round_limit=2, seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```

## Result and failure boundaries

`autom8.start()` returns `scan_object` with two completed rows. `AutoParams` supplies the required model keys; the explicit updates keep the recipe to two epochs and small networks. The held-out `x_test` and `y_test` are prepared but are not used during this scan. The program writes result and checkpoint artifacts to its experiment run directory.

The filtered dataset has two classes, matching `task="binary"` and binary cross-entropy. `AutoParams(network=False, …)` avoids external parameter-service calls. Changing a generated dictionary without retaining required model keys can break the callback; see [AutoParams](AutoParams.md). Two short trials demonstrate wiring rather than an optimized architecture.

## Read next

Return to the [walkthrough](Examples_AutoML.md) for the procedure and failure diagnosis. [AutoScan](AutoScan.md) and [AutoModel](AutoModel.md) explain the generated model boundary; [Evaluate](Evaluate.md) covers the untouched test split.
