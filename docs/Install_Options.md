# Installation

Use a virtual environment with Python 3.10–3.13. Modern Keras/TensorFlow extras require Python 3.11 or newer; Torch supports Python 3.10 or newer. Choose a framework whose supported Python versions include yours.

```sh
python -m venv .venv
. .venv/bin/activate
pip install 'talos[tensorflow]'
```

The modern minimums are TensorFlow 2.20, Keras 3.15 and Torch 2.13. The TensorFlow extra also requires patched Protobuf 6.33.5 or newer.

Framework choices:

| Extra | Purpose |
| --- | --- |
| `tensorflow` | Modern TensorFlow / tf.keras |
| `torch` | PyTorch |
| `keras,tensorflow` | Standalone Keras using TensorFlow |
| `keras,torch` | Standalone Keras using Torch; set `KERAS_BACKEND=torch` |
| `legacy-tensorflow` | Compatibility lane: TensorFlow 2.14.1, Keras 2.14 and NumPy 1.26; Python 3.10–3.11 |
| `plots` | Matplotlib experiment/training plots |
| `samplers` | Optional Chances quantum samplers |
| `test` | Acceptance tests, coverage, lint and build tools |

The legacy extra retains known upstream advisories for compatibility. Use it only for controlled existing workloads; use the modern extras for new work.

`pip install talos` installs the core and CLI without TensorFlow, Keras, Torch or plotting. Do not combine the legacy lane with modern framework extras. Upgrade with `pip install -U 'talos[your-extra]'` so dependencies are resolved together.

From this checkout:

```sh
pip install -e '.[test,plots,samplers,tensorflow,torch]'
python -m pytest -q
```
