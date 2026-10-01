# First parameter sweep

Convert an existing Keras training function into a two-candidate Talos scan. The model code remains yours; Talos selects parameter combinations, records the results and retains model assets.

## Prerequisites

Use Python 3.11–3.13 and a working environment for the [TensorFlow extra](../Backends.md). The code uses the offline scikit-learn breast-cancer dataset included in the core dependencies. Run the Python blocks in order in one session, from a writable experiment directory.

The example trains on a training split and uses a separate validation split. It demonstrates the interface, not clinical performance. A final evaluation needs another held-out split; the [typical workflow](../Scan.md#minimal-example) shows that boundary.

## Install the TensorFlow backend

```sh
pip install 'talos[tensorflow]'
```

See [installation options](../Install_Options.md) for other backends.

## Compare a model with and without Talos

First, train the Keras model directly:

```python
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

x, y = load_breast_cancer(return_X_y=True)
x_train, x_val, y_train, y_val = train_test_split(
    x[:, :8], y, test_size=.2, stratify=y, random_state=17)
normalizer = keras.layers.Normalization()
normalizer.adapt(x_train)
model = keras.Sequential([keras.layers.Input((8,)), normalizer,
                          keras.layers.Dense(12, activation='relu'),
                          keras.layers.Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer='adam')
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=1, batch_size=32, verbose=0)
```

Then, move the training code into a Talos callback and declare the parameter space:

```python
import talos


def breast_cancer_model(x_train, y_train, x_val, y_val, params):
    normalizer = keras.layers.Normalization()
    normalizer.adapt(x_train)
    model = keras.Sequential([keras.layers.Input((8,)), normalizer,
                              keras.layers.Dense(12, activation=params['activation']),
                              keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=1, batch_size=32, verbose=0)
    return history, model


scan = talos.Scan(x_train, y_train,
                 {'activation': ['relu', 'elu'], 'optimizer': ['adam']},
                 breast_cancer_model, 'minimal', x_val=x_val, y_val=y_val,
                 seed=42, disable_progress_bar=True)
```

The second block uses the imports and data split from the first. See the complete [typical example](../Examples_Typical_Code.md).

## Continue after the first scan

The most common use-case of Talos is a hyperparameter scan based on an already created Keras or TensorFlow model. In addition to the [input model](../Scan.md#input-model), a hyperparameter scan with Talos involves `talos.Scan()` command and a [parameter dictionary](../Scan.md#parameter-dictionary).

After completing an experiment, results can be analyzed and visualized. Use the results to decide whether to revise the experiment or evaluate the selected candidates on data held out from tuning. Selected models can then be used for prediction.

Talos can package trained models and experiment assets for restoration in another compatible environment; application serving remains the caller’s responsibility.

## Supported environments

- Linux, macOS or Windows system
- Python 3.10–3.13 for the core
- Optional Keras, TensorFlow/tf.keras or PyTorch; see [backend compatibility](../Backends.md)

Talos supports full-grid search, sampled search and reduction of pending candidates. It offers pseudo and quasi-random sampling, with optional external entropy providers; see [optimization strategies](../Optimization_Strategies.md).

## Check the result

`scan.data` contains two completed trial rows, one for each activation. `scan.run_dir` identifies the experiment directory. Its `results.csv` and checkpoint files retain the run’s results and pending state; see [Scan outputs](../Scan.md). Validation loss is a measurement from these short training runs, not a promised performance level.

## If the first run fails

- A missing TensorFlow import means the optional backend is absent from the active interpreter; follow [installation options](../Install_Options.md).
- Supply `x_val` and `y_val` together. Keep row counts, feature shapes and label encoding consistent with the callback.
- Return the training history and trained model in that order. A history must contain numeric scalar metrics; use the [callback contract](../Backends.md) when adapting another framework.
- Run the second Python block after the first: it reuses `keras` and the prepared arrays.

## Read next

Use the [step-by-step Iris example](../Examples_Typical.md), inspect the [Scan reference](../Scan.md), or port the callback to a [single-file definition](../SFD_and_CLI.md).
