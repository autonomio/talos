> Talos 2 preserves this interface and executes it through the shared SFD/CLI core. See [migration](Migration.md) for artifact layout, resume and corrected behavior.

# Scan

The experiment is configured and started through the `Scan()` command. All of the options effecting the experiment, other than the hyperparameters themselves, are configured through the Scan arguments. The most common use-case is where ~10 arguments are invoked.

## Minimal Example

This CPU-sized example uses standalone Keras and real Iris data. Install the optional Keras backend described in [Backends](Backends.md). Training, validation and final test data remain separate, and the scaler is fitted only on training data. Run this setup before the later fragments on this page.

```python
import keras
import numpy as np
import talos
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_all, y_all = load_iris(return_X_y=True)
x_dev, x_test, y_dev, y_test = train_test_split(
    x_all, y_all, test_size=.2, stratify=y_all, random_state=17)
x, x_val, y, y_val = train_test_split(
    x_dev, y_dev, test_size=.25, stratify=y_dev, random_state=17)
scaler = StandardScaler().fit(x)
x, x_val, x_test = [scaler.transform(a).astype('float32')
                     for a in (x, x_val, x_test)]

def input_model(x_train, y_train, x_val, y_val, params):
    model = keras.Sequential([
        keras.Input(shape=(x_train.shape[1],)),
        keras.layers.Dense(params['first_neuron'], activation=params['activation']),
        keras.layers.Dense(3, activation='softmax')])
    model.compile(optimizer=params['optimizer'],
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    return history, model

iris_model = input_model
p = {'first_neuron': [4, 8], 'activation': ['relu'], 'optimizer': ['adam'],
     'batch_size': [16], 'epochs': [2], 'hidden_layers': [1]}
scan_object = talos.Scan(x, y, p, input_model, 'iris', x_val=x_val, y_val=y_val,
                         seed=17, disable_progress_bar=True,
                         reduction_metric='val_loss', minimize_loss=True)
```

## Scan Arguments

`x`, `y`, `params`, `model`, and `experiment_name` are required to start the experiment; all other arguments are optional.

Argument | Input | Description
--------- | ------- | -----------
`x` | array or list of arrays | prediction features
`y` | array or list of arrays | prediction outcome variable
`params` | dict or ParamSpace object | the parameter dictionary or the ParamSpace object after splitting
`model` | function | the Keras, tf.keras or Torch training model as a function
`experiment_name` | str | Used for creating the experiment logging folder
`x_val` | array or list of arrays | validation data for x
`y_val` | array or list of arrays | validation data for y
`val_split` | float | validation data split ratio
`multi_input` | bool | set to True if multi-input model
`random_method` | str | the random method to be used
`seed` | int or None | Seed for random states
`performance_target` | list | A result at which point to end experiment
`fraction_limit` | float | The fraction of permutations to be processed
`round_limit` | int | Maximum number of permutations in the experiment
`time_limit` | str | Time limit for experiment in format `%Y-%m-%d %H:%M`
`boolean_limit` | function | Limit permutations based on a lambda function
`reduction_method` | str or callable | Type of reduction optimizer to be used used
`reduction_interval` | int | Number of permutations after which reduction is applied
`reduction_window` | int | the lookback window for reduction process
`reduction_threshold` | float | The threshold at which reduction is applied
`reduction_metric` | str | The metric to be used for reduction
`minimize_loss` | bool | `reduction_metric` is a loss
`disable_progress_bar` | bool | Disable live updating progress bar
`print_params` | bool | Print each permutation hyperparameters
`clear_session` | bool | Clear backend session between permutations
`save_weights` | bool | Keep model weights (increases memory pressure for large models)
`save_models` | bool | Save models in the experiment folder in local machine

`boolean_limit` is an ordinary keyword argument. A predicate returning `True` keeps a permutation; its position and line breaks do not matter:

```python

limited = talos.Scan(x, y, {**p, 'hidden_layers': [1, 2]}, input_model, 'limited',
                     x_val=x_val, y_val=y_val, disable_progress_bar=True, seed=17,
                     boolean_limit=lambda params: params['first_neuron'] * params['hidden_layers'] < 12)
```



## Scan Object Properties

Once the `Scan()` procedures are completed, an object with several useful properties is returned. The object can be used as an input to `Analyze()`, `Evaluate()`, `Predict()` and `Deploy()`, and has many properties that can be accessed directly. The namespace is strictly kept clean, so all the properties consist of meaningful contents.

In the case conducted the following experiment, we can access the properties in `scan_object` which is a python class object.

```python
scan_object = talos.Scan(x, y, model=iris_model, params=p, experiment_name='sampled',
                         x_val=x_val, y_val=y_val, fraction_limit=.5, seed=17,
                         disable_progress_bar=True)
```
<hr>

**`best_model`** picks the best model based on a given metric and returns the fitted model.

```python
scan_object.best_model(metric='val_loss', asc=True)
```
NOTE: `metric` has to be one of the metrics used in the experiment, and `asc` has to be True for the case where the metric is something to be minimized.

<hr>

**`data`** returns a pandas DataFrame with the results for the experiment together with the hyperparameter permutation details.

```python
scan_object.data
```

<hr>

**`details`** returns a pandas Series with various meta-information about the experiment.

```python
scan_object.details
```

<hr>

**`evaluate_models`** creates a new column in `scan_object.data` with result from kfold cross-evaluation.

```python
scan_object.evaluate_models(x_val=x_val,
                            y_val=y_val,
                            task='multi_class',
                            n_models=2,
                            metric='val_loss',
                            folds=5,
                            shuffle=True,
                            average='macro',
                            asc=True)
```

Argument | Description
-------- | -----------
`scan_object` | The class object returned by Scan() upon completion of the experiment.
`x_val` | Input data (features) in the same format as used in Scan(), but should not be the same data (or it will not be much of validation).
`y_val` | Input data (labels) in the same format as used in Scan(), but should not be the same data (or it will not be much of validation).
`n_models` | The number of models to be evaluated. If set to 10, then 10 models with the highest metric value are evaluated. See below.
`metric` | The metric to be used for picking the models to be evaluated.
`folds` | The number of folds to be used in the evaluation.
`shuffle` | If the data is to be shuffled or not. Set always to False for timeseries but keep in mind that you might get periodical/seasonal bias.
`average` |One of the supported averaging methods: 'binary', 'micro', or 'macro'
`asc` |Set to True if the metric is to be minimized.
`saved` | bool | if a model saved on local machine should be used
`custom_objects` | dict | if the model has a custom object, pass it here

<hr>

**`learning_entropy`** returns a pandas DataFrame with entropy measure for each permutation in terms of how much there is variation between results of each epoch in the permutation.

```python
scan_object.learning_entropy
```

<hr>

**`params`** returns a dictionary with the original input parameter ranges for the experiment.

```python
scan_object.params
```

<hr>

**`round_times`** returns a pandas DataFrame with the time when each permutation started, ended, and how many seconds it took.

```python
scan_object.round_times
```

<hr>

<hr>

**`round_history`** returns a list of dictionaries containing epoch-by-epoch data for each model.

```python
scan_object.round_history
```

<hr>

**`saved_models`** returns backend-specific in-memory model descriptions when retained: Keras JSON or Torch state dictionaries. Persisted native artifacts are available through `scan_object.artifacts`.

```python
scan_object.saved_models
```

<hr>

**`saved_weights`** returns the weights for each model.

```python
scan_object.saved_weights
```

<hr>

**`x`** returns the input data (features).

```python
scan_object.x
```

<hr>

**`y`** returns the input data (labels).

```python
scan_object.y
```

## Input Model

The input model is a callable that trains a Keras, tf.keras or Torch model. It's the model that Talos will use as the basis for the hyperparameter experiment.

#### A minimal example

```python
def input_model(x_train, y_train, x_val, y_val, params):

    model = keras.Sequential([keras.Input(shape=(4,)),
                              keras.layers.Dense(params['first_neuron'], activation=params['activation']),
                              keras.layers.Dense(3, activation='softmax')])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=params['optimizer'])
    out = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                    epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

    return out, model
```
See specific details about defining the model [here](Examples_Typical?id=defining-the-model).

#### Models with multiple inputs or outputs (list of arrays)

For both cases, pass matching nested validation data through `x_val` and `y_val`; explicit splits make alignment clear. The following fixture builds a matching model for each fragment using the Iris arrays from the minimal example:

```python
x_train, y_train = x, y
x_train_a, x_train_b = x[:, :2], x[:, 2:]
x_val_a, x_val_b = x_val[:, :2], x_val[:, 2:]
y_train_a = y_train_b = y
y_val_a = y_val_b = y_val

def make_multi_model(input_count, output_count):
    inputs = [keras.Input(shape=(2 if input_count == 2 else 4,))
              for _ in range(input_count)]
    features = keras.layers.Concatenate()(inputs) if input_count == 2 else inputs[0]
    outputs = [keras.layers.Dense(3, activation="softmax")(features)
               for _ in range(output_count)]
    model = keras.Model(inputs if input_count == 2 else inputs[0],
                        outputs if output_count == 2 else outputs[0])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model
```

For **multi-input** change `model.fit()` as highlighted below:

```python
model = make_multi_model(2, 1)
out = model.fit(x=[x_train_a, x_train_b], y=y_train,
                validation_data=([x_val_a, x_val_b], y_val), epochs=1, verbose=0)
```

For **multi-output** the same structure is expected but instead of changing the `x` argument values, now change `y`:

```python
model = make_multi_model(1, 2)
out = model.fit(x=x_train, y=[y_train_a, y_train_b],
                validation_data=(x_val, [y_val_a, y_val_b]), epochs=1, verbose=0)
```
For the case where its both **multi-input** and **multi-output** now both `x` and `y` argument values follow the same structure:

```python
model = make_multi_model(2, 2)
out = model.fit(x=[x_train_a, x_train_b], y=[y_train_a, y_train_b],
                validation_data=([x_val_a, x_val_b], [y_val_a, y_val_b]), epochs=1, verbose=0)
```


## Parameter Dictionary

The first step in an experiment is to decide the hyperparameters you want to use in the optimization process.

#### A minimal example

```python
p = {
    'first_neuron': [12, 24, 48],
    'activation': ['relu', 'elu'],
    'batch_size': [10, 20, 30]
}
```
In addition to standard Keras hyperparameters, Talos allows several extra conveniences such as the ability to include number of hidden layers in the process.

#### Supported Input Formats

Parameters may be inputted either in a list or tuple.

As a set of discrete values in a list:

```python
p = {'first_neuron': [12, 24, 48]}
```
As a range of values `(min, max, steps)`; `max` is excluded (this example gives 12 and 30):

```python
p = {'first_neuron': (12, 48, 2)}
```

For the case where a static value is preferred, but it's still useful to include it in in the parameters dictionary, use list:

```python
p = {'first_neuron': [48]}
```

#### Note on Allowed Hyperparameters

Generally speaking, whatever hyperparameters you can use in Keras, you can include in a Talos experiment as simply as including the hyperparameter label together with the desired values or the range of values in the parameter dictionary.

#### Talos Specific Parameters

In addition to common hyperparameters, Talos has several convenience functions that can be used to include otherwise unavailable parameters into experiments:

- Number of [hidden layers](Hidden_Layers.md)
- [Shape](Shapes.md) of the network
- [Normalized learning rate](Normalized_Learning_Rate.md)
