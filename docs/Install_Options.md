# Installation

Use a virtual environment with Python 3.10–3.13. Choose a framework whose supported Python versions include yours.

```sh
python -m venv .venv
. .venv/bin/activate
pip install 'talos[tensorflow]'
```

Framework choices:

| Extra | Purpose |
| --- | --- |
| `tensorflow` | Modern TensorFlow / tf.keras |
| `torch` | PyTorch |
| `keras,tensorflow` | Standalone Keras using TensorFlow |
| `keras,torch` | Standalone Keras using Torch; set `KERAS_BACKEND=torch` |
| `legacy-tensorflow` | TensorFlow 2.14.1, Keras 2.14 and NumPy 1.26; Python 3.10–3.11 |
| `plots` | Matplotlib experiment/training plots |
| `samplers` | Optional Chances quantum samplers |
| `test` | Acceptance tests, coverage, lint and build tools |

`pip install talos` installs the core and CLI without TensorFlow, Keras, Torch or plotting. Do not combine the legacy lane with modern framework extras. Upgrade with `pip install -U 'talos[your-extra]'` so dependencies are resolved together.

From this checkout:

```sh
pip install -e '.[test,plots,samplers,tensorflow,torch]'
python -m pytest -q
```
