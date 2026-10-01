# Multiple-input Keras sweep

Train a functional Keras model that receives two aligned arrays, each containing a different pair of Iris features. The same pattern applies to larger multi-input models. The [complete program](Examples_Multiple_Inputs_Code.md) combines the steps below.

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

The scikit-learn split uses one set of row indices for all aligned arrays.

### Loading data

```python
x, y = talos.templates.datasets.iris()
x_train, x_val, y_train, y_val = train_test_split(
    x.astype('float32'), y.astype('float32'), test_size=.2, random_state=17,
    stratify=y.argmax(axis=1))
```

In the case of multi-input models, the data must be split into training and validation datasets before using it in `Scan()`. `x` is expected to be a list of numpy arrays and `y` a numpy array.

**NOTE:** For full support of Talos features for multi-input models, set `Scan(...multi_input=True...)`.

### Defining the model

```python
def iris_multi(x_train, y_train, x_val, y_val, params):
    # Each input contains a different pair of measured Iris features.
    first_input = Input(shape=(2,))
    first_hidden = Dense(params['left_neurons'], activation=params['activation'])(first_input)
    second_input = Input(shape=(2,))
    second_hidden = Dense(params['right_neurons'], activation=params['activation'])(second_input)
    merged = concatenate([first_hidden, second_hidden])
    output = Dense(3, activation='softmax')(merged)
    model = Model(inputs=[first_input, second_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', talos.utils.metrics.f1score])
    out = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                    epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    return out, model
```

First, the input model must accept arguments exactly as in the example:

`def iris_multi(x_train, y_train, x_val, y_val, params):`

Even though it is a multi-input model, data can be inputted to `model.fit()` as you would otherwise do it. The multi-input part will be handled later in `Scan()` as shown below.

`model.fit(x_train, y_train, validation_data=(x_val, y_val), ...)`

The model must explicitly declare `validation_data` in `model.fit` because it is a multi-input model. Talos preserves aligned row splits across array inputs and outputs; explicit splits make this recipe easier to inspect.

`model.fit(x_train, y_train, validation_data=(x_val, y_val), ...)`

Finally, the model must `return` the `model.fit` object as well as the model itself, in the order shown:

`return out, model`

### Parameter dictionary

```python
p = {'activation': ['relu', 'elu'], 'left_neurons': [8],
     'right_neurons': [8], 'batch_size': [16], 'epochs': [2]}
```

The parameter dictionary accepts candidate lists or range tuples in the form `(min, max, number_of_values)`.

### Scan()

```python
scan_object = talos.Scan(x=[x_train[:, :2], x_train[:, 2:]], y=y_train,
                         x_val=[x_val[:, :2], x_val[:, 2:]], y_val=y_val,
                         params=p, model=iris_multi, multi_input=True,
                         experiment_name='iris_multi_input', round_limit=2,
                         seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. In the case of multi-input model, we also have to explicitly declare `x_val` and `y_val`.

Pass the same two feature arrays to prediction as you supplied during training:

```python
predictions = talos.Predict(scan_object).predict(
    [x_val[:, :2], x_val[:, 2:]], metric='val_loss', asc=True)
print(predictions.shape)  # One class-probability vector per validation row.
```

Find the description for all `Scan()` arguments [Scan arguments](Scan.md#arguments).

## Expected result

`scan_object.data` contains two completed rows. The final prediction block returns one three-class probability vector per validation row. The two input arrays preserve the order of the corresponding target rows. The run directory contains `results.csv` and checkpoint artifacts; inspect `scan_object.run_dir` for its location.

## Failure boundaries

Keep both arrays aligned and pass them in the same order to training, validation and prediction. Each input layer expects two columns; a single four-column array does not match this model. Keep `multi_input=True` when using the established multi-input facade.

If an import fails, check the active interpreter and [installation options](Install_Options.md). If a scan fails before its first trial, compare the data shapes, parameter keys and callback return with the [Scan contract](Scan.md).

## Read next

[Predict](Predict.md) describes candidate selection and input forwarding; [multiple outputs](Examples_Multiple_Outputs.md) covers aligned target lists.
