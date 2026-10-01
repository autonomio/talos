# Talos

Parameter sweeps for **Keras, TensorFlow/tf.keras and PyTorch**, with reproducible manifests, a CLI and the existing Talos Python interface.

Talos 2 adopts the general experiment infrastructure from Vaquum/Limen as an independent fork. One executor serves legacy `Scan`, native `params/prep/model` SFDs and manifest runs. Your code owns data acquisition and training.

## Install

Python 3.10–3.13 is supported by the core. Modern Keras/TensorFlow extras require Python 3.11+; Torch supports Python 3.10+.

```sh
pip install 'talos[tensorflow]'       # TensorFlow / tf.keras
pip install 'talos[torch]'            # PyTorch
pip install 'talos[keras,tensorflow]' # standalone Keras with TensorFlow
```

Standalone Keras also supports a Torch backend: install `talos[keras,torch]` and set `KERAS_BACKEND=torch` before importing Keras. The base `pip install talos` installs no DL or plotting framework. Use the `plots` extra for plotting and `samplers` for optional quantum samplers.

Existing TensorFlow 2.14 applications can use `talos[legacy-tensorflow]` on Python 3.10–3.11 with NumPy 1.26. Use a separate environment from modern backends; this compatibility lane retains known upstream advisories.

## Existing Talos code

Your five-argument callback and `Scan` call are preserved:

```python
import talos
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

features, labels = load_iris(return_X_y=True)
x, x_val, y, y_val = train_test_split(
    features, labels, test_size=.2, stratify=labels, random_state=17)
x_test = x_val
my_splits = {'x_train': x, 'y_train': y, 'x_val': x_val, 'y_val': y_val}

# Return your framework history and trained model.
def model(x_train, y_train, x_val, y_val, params):
    network = keras.Sequential([keras.layers.Input((4,)),
                                keras.layers.Dense(8, activation='relu'),
                                keras.layers.Dense(3, activation='softmax')])
    network.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    history = network.fit(x_train, y_train, validation_data=(x_val, y_val),
                          epochs=params['epochs'], batch_size=16, verbose=0)
    return history, network

scan = talos.Scan(x, y, {'epochs': [1, 2]}, model, 'my_experiment',
                 x_val=x_val, y_val=y_val, seed=42, disable_progress_bar=True)
predictions = talos.Predict(scan).predict(x_test, metric='val_loss', asc=True)
trained = scan.best_model('val_loss', asc=True)
```

`Analyze` / `Reporting`, `Evaluate`, `Deploy` / `Restore`, AutoML helpers, mutable/distributed `ParamSpace`, reducers, samplers, local strategy files, Gamify, callbacks and generators remain available. See the [migration guide](docs/Migration.md) for corrected behavior and artifact portability.

## Native SFDs

An SFD is a regular Python module with `params`, `prep` and `model`. `prep` receives caller data or loads it using your own code. `model` receives the prepared data and one parameter combination; no training base class is required.

```python
# Save as my_sfd.py.
backend = 'tensorflow'


def params():
    return {'epochs': [1, 2]}


def prep(data, round_params):
    if data is not None:
        return data
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    x, y = load_iris(return_X_y=True)
    xt, xv, yt, yv = train_test_split(x, y, test_size=.2, stratify=y, random_state=17)
    return {'x_train': xt, 'x_val': xv, 'y_train': yt, 'y_val': yv}


def model(prepared, round_params):
    from tensorflow import keras
    network = keras.Sequential([keras.layers.Input((4,)),
                                keras.layers.Dense(8, activation='relu'),
                                keras.layers.Dense(3, activation='softmax')])
    network.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    history = network.fit(prepared['x_train'], prepared['y_train'],
                          validation_data=(prepared['x_val'], prepared['y_val']),
                          epochs=round_params['epochs'], batch_size=16, verbose=0)
    return history, network
```

Save the preceding block as `my_sfd.py`. The next block uses `my_splits` and `x_test` from the Scan example.

```python
result = talos.run('my_sfd', data=my_splits, seed=42,
                   objective={'metric': 'val_loss', 'direction': 'min'})
predictions = result.predict(x_test)
```

Runnable real Iris examples are supplied for [Keras](examples/sfd/keras_sfd.py), [tf.keras](examples/sfd/tensorflow_sfd.py) and [Torch](examples/sfd/torch_sfd.py).

## Manifest and CLI

Save the following as `experiment.yaml` in this checkout.

```yaml
schema_version: "1.0"
metadata:
  name: iris
  mode: development
sfd:
  module: examples/sfd/tensorflow_sfd.py
  backend: tensorflow
  objective:
    metric: val_loss
    direction: min
  params:
    epochs: [1, 2]
uel:
  seed: 42
  search_strategy:
    type: grid
  round_limit: 4
  checkpoint_interval: 1
```

```sh
talos validate experiment.yaml
talos profile experiment.yaml
talos run --dry-run experiment.yaml
talos run --no-progress-bar experiment.yaml
run_dir=$(python -c "from pathlib import Path; print(max(Path('results/dev').glob('iris_*'), key=lambda p: p.stat().st_mtime))")
talos run --resume "$run_dir"
```

`talos new` creates a project; `init` creates editable framework templates. `commit`, `ls`, `fork`, `lineage` and `reindex` manage immutable content-addressed manifests. `backup` snapshots a project to its configured Git remote. See [SFD and CLI usage](docs/SFD_and_CLI.md).

Runs contain metadata, exact trial identities, histories, native trained artifacts, CSV results, queue/control checkpoints and an intervention audit. Callable values use importable references; opaque data supports an explicit fingerprint. Resume validates code, data/splits, configuration and environment. Completed trials are restored from saved artifacts.

## Development

```sh
pip install -e '.[test,plots,samplers,tensorflow,torch]'
python -m pytest -q
ruff check talos tests/test_*.py
python -m build
```

The maintained suite uses real Iris and breast cancer fixtures, three framework artifact round trips, legacy compatibility, live controls, manifests and interrupted runs. Historical tests under `tests/commands` remain reference examples; the maintained acceptance suite replaces their obsolete dependency assumptions.

Talos is MIT licensed. [NOTICE](NOTICE) records Limen attribution and the fork baseline. [CONTRIBUTING.md](CONTRIBUTING.md) describes verification and maintenance.
