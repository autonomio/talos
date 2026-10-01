# Installation

Install Talos into an isolated environment, then choose the framework extra your model uses. The package exports the Python API and the `talos` command; optional frameworks are loaded only when required. This page owns installation choices rather than training or CLI configuration.

Prerequisites are Python 3.10–3.13 and pip with access to a package index containing the required distributions. Modern Keras/TensorFlow extras require Python 3.11 or newer; Torch supports Python 3.10 or newer. Choose a framework whose supported Python and platform versions include yours. The activation command below is for a POSIX shell; on Windows use the environment's activation script for that shell.

## Install a framework extra

If your package index does not yet offer the required version and extras, use the checkout installation below.

```sh
python -m venv .venv
. .venv/bin/activate
pip install 'talos[tensorflow]'
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
| `legacy-tensorflow` | Compatibility lane: TensorFlow 2.14.1, Keras 2.14 and NumPy 1.26; Python 3.10–3.11 |
| `plots` | Matplotlib experiment/training plots |
| `samplers` | Optional Chances quantum samplers |
| `test` | Acceptance tests, coverage, lint and build tools |

The legacy extra retains known upstream advisories for compatibility. Use it only for controlled existing workloads; use the modern extras for new work.

`pip install talos` installs the core and CLI without TensorFlow, Keras, Torch or plotting. Do not combine the legacy lane with modern framework extras. Upgrade with `pip install -U 'talos[your-extra]'` so dependencies are resolved together.

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
