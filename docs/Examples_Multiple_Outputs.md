# Multiple-output Keras sweep

This example predicts two real outcomes from the Wisconsin Breast Cancer dataset: diagnosis and measured mean radius. The same multi-output pattern can model experiment outcomes, as in the original Telco Churn illustration of using hyperparameter optimization data to optimize hyperparameter optimization.

The [complete program](Examples_Multiple_Outputs_Code.md) combines the steps below.

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
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
x, diagnosis = load_breast_cancer(return_X_y=True)
# Predict diagnosis and the separately measured mean radius, without input leakage.
radius = x[:, 0:1].astype('float32') / 30
features = x[:, 1:].astype('float32')
x_train, x_val, diagnosis_train, diagnosis_val, radius_train, radius_val = train_test_split(
    features, diagnosis, radius, train_size=144, test_size=36,
    stratify=diagnosis, random_state=17)
scaler = StandardScaler().fit(x_train)
x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)
y_train = [diagnosis_train, radius_train]
y_val = [diagnosis_val, radius_val]
```

In the case of multi-output models, the data must be split into training and validation datasets before using it in `Scan()`. `x` is expected to be a numpy array, and `y` a list of numpy arrays.

### Defining the model

```python
def breast_cancer_multi(x_train, y_train, x_val, y_val, params):
    input_layer = Input(shape=(x_train.shape[1],))
    hidden = Dense(params['neurons'], activation=params['activation'])(input_layer)
    diagnosis = Dense(1, activation='sigmoid', name='diagnosis')(hidden)
    radius = Dense(1, name='radius')(hidden)
    model = Model(inputs=input_layer, outputs=[diagnosis, radius])
    model.compile(optimizer='adam',
                  loss={'diagnosis': 'binary_crossentropy', 'radius': 'mse'},
                  metrics={'diagnosis': ['accuracy', talos.utils.metrics.f1score],
                           'radius': ['mae']})
    out = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                    epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    return out, model
```

First, the input model must accept arguments exactly as in the example:

`def breast_cancer_multi(x_train, y_train, x_val, y_val, params):`

Even though it is a multi-output model, data can be inputted to `model.fit()` as you would otherwise do it. The multi-output part will be handled later in `Scan()` as shown below.

`model.fit(x_train, y_train, validation_data=(x_val, y_val), ...)`

The model must explicitly declare `validation_data` in `model.fit` because it is a multi-output model. Talos preserves aligned row splits across array inputs and outputs; explicit splits make this recipe easier to inspect.

`model.fit(x_train, y_train, validation_data=(x_val, y_val), ...)`

Finally, the model must `return` the `model.fit` object as well as the model itself, in the order shown:

`return out, model`

### Parameter dictionary

```python
p = {'activation': ['relu', 'elu'], 'neurons': [8],
     'batch_size': [16], 'epochs': [2]}
```

The parameter dictionary accepts candidate lists or range tuples in the form `(min, max, number_of_values)`.

### Scan()

```python
scan_object = talos.Scan(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                         params=p, model=breast_cancer_multi,
                         experiment_name='breast_cancer_multi_output', round_limit=2,
                         seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. In the case of multi-output model, we also have to explicitly declare `x_val` and `y_val`.

Pass `y_train` and `y_val` as aligned lists. The trained model produces one output array per target:

```python
predictions = talos.Predict(scan_object).predict(x_val, metric='val_loss', asc=True)
print([output.shape for output in predictions])  # Diagnosis and radius arrays.
```

Find the description for all `Scan()` arguments [Scan arguments](Scan.md#arguments).

## Expected result

`scan_object.data` contains two completed rows with aggregate and output-specific metrics from the training history. Prediction returns a list of two arrays: diagnosis probabilities and normalized radius estimates. Both arrays have one row per validation example. The run directory contains `results.csv` and checkpoint artifacts; inspect `scan_object.run_dir` for its location.

## Failure boundaries

Keep both target arrays aligned with the features and in the model’s output order. The radius target is the separately measured first feature, removed from the input to avoid direct leakage. Diagnosis and radius have different losses and units; aggregate validation loss alone does not establish either output’s scientific usefulness.

If an import fails, check the active interpreter and [installation options](Install_Options.md). If a scan fails before its first trial, compare the data shapes, parameter keys and callback return with the [Scan contract](Scan.md).

## Read next

[Evaluate](Evaluate.md) explains held-out evaluation; [Predict](Predict.md) describes model selection and returned predictions.
