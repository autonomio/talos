# Talos 2

Start with the [current project overview](../README.md), [migration guide](Migration.md) and [SFD/CLI guide](SFD_and_CLI.md). The reference chapters below retain the established Talos workflow; the migration guide records corrected contracts and backend requirements.

# Quick start

```sh
pip install 'talos[tensorflow]'
```
See [here](Install_Options.md) for more options.


# Minimal Example

Your Keras model **WITHOUT** Talos:

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
Your Keras model **WITH** Talos:


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
The second block uses the imports and data split from the first. See the complete [typical example](Examples_Typical_Code.md).

# Typical Use-cases

The most common use-case of Talos is a hyperparameter scan based on an already created Keras or TensorFlow model. In addition to the [input model](Scan.md#input-model), a hyperparameter scan with Talos involves `talos.Scan()` command and a [parameter dictionary](Scan.md#params).

After completing an experiment, results can be analyzed and visualized. Once a decision have been made if a) the experiment should be reconfigured and continue or b) sufficient level of performance have already been found, model candidates can be automatically evaluated and used for prediction.

Talos also supports easy deployment of models and experiment assets from the experiment environment to production or other systems.

# System Requirements

- Linux, Mac OSX or Windows system
- Python 3.10–3.13 for the core
- Optional Keras, TensorFlow/tf.keras or PyTorch; see [backend compatibility](Backends.md)

Talos incorporates grid, random, and probabilistic hyperparameter optimization strategies, with focus on maximizing the flexibility, efficiency, and result of random strategy. Talos users benefit from access to pseudo, quasi, true, and quantum random methods.
