# Installation

Install Talos into an isolated environment, then choose the framework extra your model uses. The package exports the Python API and the `talos` command; optional frameworks are loaded only when required. This page owns installation choices rather than training or CLI configuration.

Prerequisites are Python 3.10–3.13 and pip with access to a package index containing the required distributions. Modern Keras/TensorFlow extras require Python 3.11 or newer; Torch supports Python 3.10 or newer. Choose a framework whose supported Python and platform versions include yours. The activation command below is for a POSIX shell; on Windows use the environment's activation script for that shell.

## Install a framework extra

Talos 2 is available on PyPI. Install the framework extra you need, or use the checkout installation below. For research, pin the exact Talos version; source installations should use a full reviewed commit.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install 'talos[tensorflow]'
```

The modern minimums are TensorFlow 2.20, Keras 3.15 and Torch 2.13. The TensorFlow extra also requires patched Protobuf 6.33.5 or newer.

## Available extras

Framework choices:

| Extra | Purpose |
| --- | --- |
| `tensorflow` | Modern TensorFlow / tf.keras |
| `torch` | PyTorch |
| `keras,tensorflow` | Standalone Keras using TensorFlow |
| `keras,torch` | Standalone Keras using Torch; set `KERAS_BACKEND=torch` |
| `legacy-tensorflow` | TensorFlow 2.14.1 with owned Keras/Protobuf security backports and NumPy 1.26 |
| `plots` | Matplotlib experiment/training plots |
| `samplers` | Compatibility extra; remote samplers use the core's verified HTTPS clients |
| `test` | Acceptance tests, coverage, lint and build tools |

The legacy extra requires the owned security wheels described in [Security backports](Developer/Security-Backports.md). Build or verify and install those wheels before selecting this extra. This lane supports Python 3.10.12+ or 3.11.4+; do not combine it with modern extras. Upstream advisory lookups remain visible and require verified repairs or explicit absence evidence. Use modern extras for new work.

`python -m pip install talos` installs the released core and CLI without TensorFlow, Keras, Torch or plotting. Change the extra in the installation command to choose a backend. Do not combine the legacy lane with modern framework extras. Resolve framework and Talos upgrades together in a fresh environment.

## Established Talos 1.x

The last published old-generation release is Talos 1.4. In a separate Python 3.10.12+ or 3.11.4+ environment, install the owned legacy framework wheels from [Security backports](Developer/Security-Backports.md), then install `python -m pip install 'talos==1.4' 'tensorflow==2.14.1' 'numpy==1.26.4' 'ipython<9'`. The IPython constraint preserves compatibility with its `kerasplotlib` dependency. The official unchanged wheel is verified on Python 3.11 with TensorFlow 2.14.1, Keras 2.14.0, NumPy 1.26.4 and IPython 8.39.0.

Talos 1.x will remain supported at least until 2028. The owned framework backports preserve the unchanged Talos wheel and old callback interface. Plain upstream Keras 2.14 and Protobuf 4.25.9 still carry known findings; do not infer a repair from the Talos support commitment. The `legacy-tensorflow` extra above runs historical framework callbacks on Talos 2; it does not install Talos 1.4.

## Install this checkout

For Linux CPU use, first install the [CPU Torch build](#linux-cpu-verification) in your environment. Run these commands from the repository root to install the working source with development and framework extras, then execute the acceptance suite:

```sh
pip install -e '.[test,plots,samplers,tensorflow,torch]'
python -m pytest -q
```

## Verify installation and troubleshoot

A successful install provides `import talos` and `talos --help`; plain core installation does not provide a deep-learning framework. `talos.__version__` reports the installed package version. From a checkout, the acceptance command above should complete without failures in an environment with its declared extras.

A resolver error or missing wheel usually means the selected framework does not support that Python/platform combination or the requested extras conflict. Create a fresh environment for a different compatibility lane rather than mixing TensorFlow 2.14 with modern Keras. Keras backend selection must happen before importing Keras; set `KERAS_BACKEND=torch` before starting a process using the Torch Keras backend.

An editable install reads the working tree, so experiments can change behavior when that tree changes. For reproducible research, retain exact installed versions together with the run's manifest and provenance; editable installation alone is not a release artifact. GPU drivers and framework-specific accelerator setup remain external to Talos installation.

The exact dependency bounds are maintained in [pyproject.toml](../pyproject.toml). [Maintenance](Maintenance.md) records the supported compatibility lanes and verification process.

## Linux CPU verification

For Linux CPU workloads, install a Torch version within Talos's declared bounds using the CPU compute platform in [PyTorch's installation selector](https://pytorch.org/get-started/locally/) before installing the Talos Torch extra or the combined development extras. The Linux verification lanes use official CPU wheels and record their exact versions and hashes in `requirements/ci/`.

The combined TensorFlow/PyTorch test environment exposed a native Triton import crash with the default CUDA-enabled Torch build on a CPU runner. The verified CPU wheels avoid that stack. These results establish CPU compatibility; accelerator libraries and driver compatibility require verification on their target hardware.

## Read next

Run the [quickstart](Guides/Quickstart.md), choose a [backend](Backends.md), or create an [SFD and CLI experiment](SFD_and_CLI.md).
