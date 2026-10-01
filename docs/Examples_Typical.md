# Typical Keras sweep

Train two small Iris classifiers while varying their hidden-layer activation. This is the starting recipe for adapting an existing Keras model; the [complete program](Examples_Typical_Code.md) combines the same steps.

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
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, concatenate
```

### Loading data

```python
x, y = talos.templates.datasets.iris()
x_train, x_val, y_train, y_val = train_test_split(
    x.astype('float32'), y.astype('float32'), test_size=.2, random_state=17,
    stratify=y.argmax(axis=1))
```

`x` and `y` are expected to be either numpy arrays or lists of numpy arrays.

### Defining the model

```python
def iris_model(x_train, y_train, x_val, y_val, params):
    model = Sequential([Input(shape=(4,)),
                        Dense(params['first_neuron'], activation=params['activation']),
                        Dense(3, activation='softmax')])
    model.compile(optimizer=params['optimizer'], loss=params['losses'],
                  metrics=['accuracy', talos.utils.metrics.f1score])
    out = model.fit(x_train, y_train, batch_size=params['batch_size'],
                    epochs=params['epochs'], validation_data=(x_val, y_val),
                    verbose=0)
    return out, model
```

First, the input model must accept arguments exactly as in the example:

`def iris_model(x_train, y_train, x_val, y_val, params):`

Second, the model must explicitly declare `validation_data` in `model.fit`:

`model.fit(x_train, y_train, validation_data=(x_val, y_val), ...)`
Finally, the model must `return` the `model.fit` object as well as the model itself, in the order shown:

`return out, model`

### Parameter dictionary

```python
p = {'activation': ['relu', 'elu'],
     'first_neuron': [8], 'optimizer': ['adam'],
     'losses': ['categorical_crossentropy'],
     'batch_size': [16], 'epochs': [2]}
```

The parameter dictionary accepts candidate lists or range tuples in the form `(min, max, number_of_values)`.

### Scan()

```python
scan_object = talos.Scan(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                         model=iris_model, params=p, experiment_name='iris',
                         round_limit=2, seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. Find the description for all `Scan()` arguments [Scan arguments](Scan.md#arguments).

## Expected result

`scan_object.data` contains two completed rows. Each row records the selected activation and final training/validation metrics. The model returns a probability vector for each of the three Iris classes. The run directory contains `results.csv` and checkpoint artifacts; inspect `scan_object.run_dir` for its location.

## Failure boundaries

The Iris labels are one-hot vectors, so the output has three units and uses categorical cross-entropy. Keep this encoding, the output shape and the loss consistent when replacing the dataset.

If an import fails, check the active interpreter and [installation options](Install_Options.md). If a scan fails before its first trial, compare the data shapes, parameter keys and callback return with the [Scan contract](Scan.md).

## Read next

[Analyze results](Analyze.md), then [evaluate candidates](Evaluate.md) on data held out from tuning.
