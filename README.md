# Talos

Parameter sweeps for **Keras, TensorFlow/tf.keras and PyTorch**, with reproducible manifests, a CLI and the existing Talos Python interface.

Talos 2 adopts the general experiment infrastructure from Vaquum/Limen as an independent fork. One executor serves legacy `Scan`, native `params/prep/model` SFDs and manifest runs. Your code owns data acquisition and training.

## Install

Python 3.10–3.13 is supported by the core; backend support follows the selected framework.

```sh
pip install 'talos[tensorflow]'       # TensorFlow / tf.keras
pip install 'talos[torch]'            # PyTorch
pip install 'talos[keras,tensorflow]' # standalone Keras with TensorFlow
```

Standalone Keras also supports a Torch backend: install `talos[keras,torch]` and set `KERAS_BACKEND=torch` before importing Keras. The base `pip install talos` installs no DL or plotting framework. Use the `plots` extra for plotting and `samplers` for optional quantum samplers.

Existing TensorFlow 2.14 applications can use `talos[legacy-tensorflow]` on Python 3.10–3.11 with NumPy 1.26. Use a separate environment from modern backends.

## Existing Talos code

Your five-argument callback and `Scan` call are preserved:

```python
import talos

# Return your framework history and trained model.
def model(x_train, y_train, x_val, y_val, params):
    network = build_your_model(params)
    history = network.fit(x_train, y_train,
                          validation_data=(x_val, y_val),
                          epochs=params['epochs'])
    return history, network

scan = talos.Scan(x, y, {'epochs': [5, 10]}, model, 'my_experiment')
predictions = talos.Predict(scan).predict(x_test, metric='val_loss', asc=True)
trained = scan.best_model('val_loss', asc=True)
```

`Analyze` / `Reporting`, `Evaluate`, `Deploy` / `Restore`, AutoML helpers, mutable/distributed `ParamSpace`, reducers, samplers, local strategy files, Gamify, callbacks and generators remain available. See the [migration guide](docs/Migration.md) for corrected behavior and artifact portability.

## Native SFDs

An SFD is a regular Python module with `params`, `prep` and `model`. `prep` receives caller data or loads it using your own code. `model` receives the prepared data and one parameter combination; no training base class is required.

```python
# my_sfd.py

def params():
    return {'epochs': [5, 10]}


def prep(data, round_params):
    return data  # or call your own loader and prepare your own splits


def model(prepared, round_params):
    network = build_your_model(round_params)
    history = train_your_model(network, prepared, round_params)
    return history, network
```

```python
result = talos.run('my_sfd', data=my_splits, seed=42,
                   objective={'metric': 'val_loss', 'direction': 'min'})
predictions = result.predict(x_test)
```

Runnable real Iris examples are supplied for [Keras](examples/sfd/keras_sfd.py), [tf.keras](examples/sfd/tensorflow_sfd.py) and [Torch](examples/sfd/torch_sfd.py).

## Manifest and CLI

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
talos run --resume results/dev/<run-directory>
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
