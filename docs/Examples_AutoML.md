# Bounded AutoML sweep

Run a two-trial binary Iris search using Talos `AutoScan`, `AutoParams` and built-in model architectures. The [complete program](Examples_AutoML_Code.md) combines the steps below.

## Prerequisites

Use Python 3.11–3.13 with the [TensorFlow extra](Backends.md) (`talos[tensorflow]`) installed in the active interpreter. The scikit-learn dataset is available offline through the core dependencies. Run the Python blocks in order, in one session, from a writable experiment directory. These bounded training runs demonstrate the interface; they do not establish clinical or generalization performance.

## Procedure

1. Import the libraries for this recipe.
2. Prepare aligned training and validation data.
3. Define the callback, or select the built-in AutoML model.
4. Declare the parameter candidates.
5. Run the bounded Scan configuration and inspect its completed rows.

### Imports

```python
import talos
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### Loading data

```python
x, y = load_iris(return_X_y=True)
x, y = x[y < 2].astype('float32'), y[y < 2]
x, x_test, y, y_test = train_test_split(x, y, test_size=.1, stratify=y, random_state=17)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, stratify=y, random_state=17)
scaler = StandardScaler().fit(x_train)
x_train, x_val, x_test = [scaler.transform(part) for part in (x_train, x_val, x_test)]
```

`x` and `y` are expected to be either numpy arrays or lists of numpy arrays and same applies for the case where `x_train`, `y_train`, `x_val`, `y_val` is used instead.

### Defining the model

This recipe uses the built-in model rather than defining a callback. `talos.autom8.AutoModel()` is used behind the scenes, where several model architectures fully wired for Talos are found. We simply initiate the `AutoScan()` object first:

```python
autom8 = talos.autom8.AutoScan(task='binary', experiment_name='iris_automl', max_param_values=2)
# AutoParams supplies all required keys; bound this educational run explicitly.
p = talos.autom8.AutoParams(task='binary', network=False, resample_params=1).params
p.update({'epochs': [2], 'first_neuron': [8], 'hidden_layers': [0],
          'batch_size': [16], 'dropout': [0.], 'losses': ['binary_crossentropy'],
          'activation': ['relu', 'elu']})
```

### Parameter dictionary

The complete dictionary is generated with `AutoParams()`. The example limits epochs and network size to keep the first run small; expand these values for research.

### Scan()

Start the scan by calling the `AutoScan.start()` method.

```python
scan_object = autom8.start(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                           params=p, round_limit=2, seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```

We pass data here just like we would do it in `Scan()` normally. Also, you are free to use any of the `Scan()` arguments here to configure the experiment. Find the description for all `Scan()` arguments [Scan arguments](Scan.md#arguments).

## Expected result

`autom8.start()` returns `scan_object` with two completed rows. `AutoParams` supplies the required model keys; the explicit updates keep the recipe to two epochs and small networks. The held-out `x_test` and `y_test` are prepared but are not used during this scan. The run directory contains `results.csv` and checkpoint artifacts; inspect `scan_object.run_dir` for its location.

## Failure boundaries

The filtered dataset has two classes, matching `task="binary"` and binary cross-entropy. `AutoParams(network=False, …)` avoids external parameter-service calls. Changing a generated dictionary without retaining required model keys can break the callback; see [AutoParams](AutoParams.md). Two short trials demonstrate wiring rather than an optimized architecture.

If an import fails, check the active interpreter and [installation options](Install_Options.md). If a scan fails before its first trial, compare the data shapes, parameter keys and callback return with the [Scan contract](Scan.md).

## Read next

[AutoScan](AutoScan.md) and [AutoModel](AutoModel.md) explain the generated model boundary; [Evaluate](Evaluate.md) covers the untouched test split.
