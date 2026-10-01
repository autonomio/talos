# Scan

`talos.Scan` configures and runs a parameter-sweep experiment synchronously through a five-argument training callback. Import it from `talos`. Hyperparameter candidates belong in `params`; execution, sampling, reduction and persistence options belong in Scan arguments.

Talos 2 preserves this Python interface over the shared [SFD and CLI core](SFD_and_CLI.md). The caller owns data acquisition, model architecture, training and scientific evaluation. Scan owns candidate selection, per-trial execution, result logging, model artifacts and checkpoints. See [migration](Migration.md) for artifact layout, resume and corrected behavior.

## Minimal example

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

<a id="scan-arguments"></a>
<a id="the-details"></a>

## Arguments

`x`, `y`, `params`, `model`, and `experiment_name` are required to start the experiment; all other arguments are optional.

| Argument | Input | Default | Description |
| --- | --- | --- | --- |
| `x` | array or list of arrays | required | Training features; list inputs require `multi_input=True`. |
| `y` | array or list of arrays | required | Training targets, aligned with features. |
| `params` | dict or ParamSpace | required | Candidate dictionary or a prepared/sharded [ParamSpace](Parallelism.md). |
| `model` | callable | required | Keras, tf.keras or Torch training callback. |
| `experiment_name` | str | required | Parent logging folder name unless `experiment_dir` is supplied. |
| `x_val` | array or list of arrays | `None` | Explicit validation features; supply with `y_val`. |
| `y_val` | array or list of arrays | `None` | Explicit validation targets; supply with `x_val`. |
| `val_split` | float | `0.3` | Validation fraction used when no explicit validation pair is supplied. |
| `multi_input` | bool | `False` | Treat list feature inputs as aligned model inputs. |
| `random_method` | str | `'uniform_mersenne'` | Legacy sampling method applied when a fraction or round limit selects candidates. |
| `seed` | int or None | `None` | Split/sampler and per-trial random seed; backend deterministic operations remain caller-controlled. |
| `performance_target` | list or None | `None` | `[metric, threshold, minimize]`; stop when the latest trial reaches the threshold. |
| `fraction_limit` | float or None | `None` | Fraction of Cartesian combinations to sample before training; takes precedence over `round_limit` for legacy candidate selection. |
| `round_limit` | int or None | `None` | Number of Cartesian combinations to sample when no fraction limit is present. |
| `time_limit` | str or None | `None` | Local wall-clock deadline in `%Y-%m-%d %H:%M` format; checked between trials. |
| `boolean_limit` | callable or None | `None` | Predicate applied to candidate dictionaries; `True` keeps a permutation. |
| `reduction_method` | str, callable or None | `None` | [Reduction optimizer](Optimization_Strategies.md) or custom reducer. |
| `reduction_interval` | int | `50` | Completed-trial cadence for correlation/tree/custom reduction. Local strategy and Gamify run at their own per-trial boundary. |
| `reduction_window` | int | `20` | Lookback window used by reduction analysis. |
| `reduction_threshold` | float | `0.2` | Reducer-specific threshold. |
| `reduction_metric` | str | `'val_acc'` | Result metric for reduction and the default run objective. |
| `minimize_loss` | bool | `False` | Minimize the reduction metric and default objective when `True`. |
| `disable_progress_bar` | bool | `False` | Disable live round progress. |
| `print_params` | bool | `False` | Print each trial's hyperparameters. |
| `clear_session` | bool | `True` | Run backend session/device cleanup after each trial. |
| `save_weights` | bool | `True` | Persist native artifacts and retain compatibility weights/descriptions when `save_models=False`; increases memory use. |
| `save_models` | bool | `False` | Persist native artifacts without retaining live models by default. |
| `**options` | keywords | empty | Additional shared-core options described below. |

`boolean_limit` is an ordinary keyword argument. A predicate returning `True` keeps a permutation; its position and line breaks do not matter:

```python

limited = talos.Scan(x, y, {**p, 'hidden_layers': [1, 2]}, input_model, 'limited',
                     x_val=x_val, y_val=y_val, disable_progress_bar=True, seed=17,
                     boolean_limit=lambda params: params['first_neuron'] * params['hidden_layers'] < 12)
```

### Shared-core keyword options

| Option | Default | Effect |
| --- | --- | --- |
| `experiment_dir` | `None` | Use an explicit run directory instead of creating one under `experiment_name`. A new run requires an empty destination. |
| `backend` | `None` | Infer the framework from the returned model, or supply a framework hint; see [Backends](Backends.md). |
| `model_factory` | `None` | Torch reconstruction factory, optionally paired with constructor configuration. |
| `objective` | derived | Explicit `{'metric': name, 'direction': 'min' or 'max'}`; otherwise derived from `reduction_metric` and `minimize_loss`. |
| `retain_models` | derived | Retain live models; defaults to `not save_models and save_weights`. |
| `output_format` | `'csv'` | `'csv'` or `'parquet'` for the additional result projection; the CSV compatibility result remains available. |
| `checkpoint_interval` | `1` | Completed-trial checkpoint cadence. |
| `feedback_interval` | `100` | Completed-trial cadence for native feedback processing. |

The full shared execution interface is described in [SFD and CLI](SFD_and_CLI.md). Resume uses recorded source, data, environment and candidate identities; see [migration](Migration.md) before setting `resume=True` with an explicit directory.

<a id="scan-object-properties"></a>

## Result properties

Construction returns a Scan object after execution completes or pauses. It can be passed to [Analyze](Analyze.md), [Evaluate](Evaluate.md), [Predict](Predict.md) and [Deploy](Deploy.md). The example below samples half of the parameter space; the subsequent properties refer to that returned object.

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

**`evaluate_models`** adds held-out F1 or MAE mean/std columns to `scan_object.data`. Each selected model is scored on subsets without being retrained; see [Evaluate](Evaluate.md) for signatures and scientific score semantics.

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

| Argument | Description |
| --- | --- |
| `x_val`, `y_val` | Held-out features and labels in the fitted model's expected structure. |
| `task` | Classification or continuous score choice. |
| `n_models` | Maximum number of candidates selected by the scan metric; default `10`. |
| `metric` | Existing result column for candidate selection; default `'val_acc'`. |
| `folds` | Held-out scoring subsets; default `5`, without retraining. |
| `shuffle` | Default `True`; disable when row order must be preserved. |
| `average` | F1 averaging choice, with task-specific defaults. |
| `asc` | Default `False`; use `True` to minimize the selection metric. |
| `saved` | Load a persisted model rather than using a retained live model. |
| `custom_objects` | Keras objects required for reconstruction. |
| `model_factory` | Torch constructor factory when needed. |
| `seed` | Seed for shuffled held-out subset assignment. |

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

## Input model

The input model is a callable that trains a Keras, tf.keras or Torch model. It's the model that Talos will use as the basis for the hyperparameter experiment.

### A minimal example

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

See [defining the model](Examples_Typical.md#defining-the-model) for the callback walkthrough.

<a id="models-with-multiple-inputs-or-outputs-list-of-arrays"></a>

### Multiple inputs and outputs

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

<a id="params"></a>

## Parameter dictionary

The first step in an experiment is to decide the hyperparameters you want to use in the optimization process.

### Candidate dictionary example

```python
p = {
    'first_neuron': [12, 24, 48],
    'activation': ['relu', 'elu'],
    'batch_size': [10, 20, 30]
}
```

In addition to standard Keras hyperparameters, Talos allows several extra conveniences such as the ability to include number of hidden layers in the process.

### Supported input formats

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

<a id="note-on-allowed-hyperparameters"></a>

### Allowed hyperparameters

Generally speaking, whatever hyperparameters you can use in Keras, you can include in a Talos experiment as simply as including the hyperparameter label together with the desired values or the range of values in the parameter dictionary.

<a id="talos-specific-parameters"></a>

### Talos convenience parameters

In addition to common hyperparameters, Talos has several convenience functions that can be used to include otherwise unavailable parameters into experiments:

- Number of [hidden layers](Hidden_Layers.md)
- [Shape](Hidden_Layers.md#shapes) of the network
- [Normalized learning rate](Learning_Rate_Normalizer.md)

## Artifacts, retention and failures

The minimal example completes two Iris trials. `.data` has one row per completed trial, final epoch metrics, candidate values, UTC start/end timestamps, elapsed seconds (`duration` and `execution_time`), epoch count and stable trial/parameter identifiers. Metric/parameter name collisions use `.parameter_columns` aliases rather than replacing measured values.

`.run_dir` is an absolute Path. `results.csv` is the compatibility result table; `round_data.jsonl`, `metadata.json` and `checkpoint.json` retain trial records, provenance and resumable state. Model descriptors are exposed through `.artifacts`. Either `save_weights=True` or `save_models=True` writes backend-native model artifacts for recoverable returned models. With both flags disabled, model selection needs an explicitly retained live model; a later process cannot recover an unpersisted model.

The default objective names `val_acc`, while a callback compiled with `'accuracy'` commonly emits `val_accuracy`. Set `reduction_metric` or `objective` to a column actually emitted by the callback, and set its direction explicitly. `best_model(metric='val_acc', asc=False, ...)` preserves the historical default; pass a metric and ascending direction for unambiguous selection. `best_model(metric=None)` uses the run objective.

Scan rejects a noncallable model, a parameter object other than dict/ParamSpace, an unpaired validation argument, and list feature inputs without `multi_input=True`. Automatic validation splitting requires `0 < val_split < 1` and nonempty train/validation partitions. Candidate values must be nonempty lists or valid `(start, end, steps)` tuples; integer range conversion can deduplicate widths. Fraction/round sampling that selects fewer than one permutation fails. Optional quantum/ambience samplers require the `samplers` extra and their external services.

Training errors propagate after checkpointing committed trials. A pause or interrupt keeps completed trials and the pending candidate for compatible resume; it does not promise recovery of an interrupted model's partial training. Seeded runs record their seeds, but hardware kernels, framework configuration, random crypto/external samplers and caller side effects can still prevent bitwise reproducibility.

## Read next

Inspect results with [Analyze](Analyze.md), score models with [Evaluate](Evaluate.md), choose [optimization strategies](Optimization_Strategies.md), or port the callback using [migration](Migration.md).
