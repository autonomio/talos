# Backends

Talos runs standalone Keras, TensorFlow/tf.keras and native PyTorch callbacks through the same parameter-sweep, result and artifact interfaces. The core does not import a framework until training, prediction, serialization or a framework-specific helper requires it.

This page covers framework selection and normalized callback results. [Installation](Install_Options.md) owns dependency extras, [Scan](Scan.md) owns the five-argument Python interface, and [SFD and CLI](SFD_and_CLI.md) owns structured experiments. Dataset acquisition and framework-specific training loops remain caller responsibilities.

## TensorFlow

Install `talos[tensorflow]` on Python 3.11 or newer (TensorFlow 2.20+, Keras 3.15+). Existing five-argument callbacks returning `(history, model)` work unchanged. The separate `legacy-tensorflow` lane preserves TensorFlow 2.14 / Keras 2.14 applications on Python 3.10–3.11. Install the [owned security backports](Developer/Security-Backports.md) before this compatibility extra. Original upstream advisories remain in the audit with exact repair or absence evidence; modern extras are the default for new work.

## Keras

On Python 3.11 or newer, install `talos[keras,tensorflow]`, or `talos[keras,torch]` with `KERAS_BACKEND=torch`. Set the backend before importing Keras. Native `.keras` artifacts retain trained weights; pass `custom_objects` for custom layers/metrics when required.

## PyTorch

Install `talos[torch]` (Torch 2.13+) on Python 3.10 or newer. Return `(history, model)` or a structured SFD result. The documented historical return of a network carrying `.history` plus its parameter iterator is normalized to the trained network.

For portable restoration, provide an importable factory and constructor configuration, as demonstrated in [the Torch SFD](../examples/sfd/torch_sfd.py). Native artifacts use `state_dict`; nested class instances need an explicit restoration factory. Prediction preserves the network's existing training mode around inference.

See [migration](Migration.md) for archive and resume behavior.

## Callback and adapter interfaces

Import the user-facing entry points from `talos`. A Scan callback accepts `(x_train, y_train, x_val, y_val, params)` and returns `(history, fitted_model)`. History may be a framework History object or a dictionary mapping metric names to sequences of numeric scalar values. Each trial's summary uses the final value of each nonempty sequence; `.round_history` retains every supplied epoch value.

SFD model functions may instead return a dictionary with `metrics`, `history`, `model`, `backend` and `model_factory` fields. Metric-only experiments are valid, but predicting or deploying requires a trained model. Invalid histories, non-string metric names and nonnumeric metric values raise `TypeError` before the trial is committed.

For custom integrations, `talos.backends.backend_for(model=None, backend=None)` returns the inferred or named adapter, and `normalise_result(output, backend=None, model_factory=None)` validates callback output. Supported framework hints are `keras`, `tensorflow` and `torch`, with `tf`/`tf.keras` and `pytorch` aliases. A hint can be supplied through Scan's `backend` keyword or an SFD's `backend` attribute. Unsupported hints raise `ValueError`.

| Framework | Prediction | Native artifact and restoration |
| --- | --- | --- |
| Standalone Keras | Calls `model.predict()` with `verbose=0` by default. | `.keras` with an integrity digest; loads with `compile=False`. Custom objects may be supplied explicitly. |
| TensorFlow/tf.keras | Same Keras prediction contract. | `.keras`, plus historical JSON/H5 compatibility in [Restore](Restore.md). |
| Native Torch | Uses `model.predict()` if provided; otherwise calls the module under evaluation mode and `torch.no_grad()`. | CPU-copied `.pt` state dictionary plus constructor configuration and factory identity. |

Torch arrays are moved to the model's device and converted to `float32`; existing tensors retain their dtype. List/tuple inputs are positional module inputs and dictionaries are keyword inputs. Standard module outputs are converted back to NumPy, including nested structures. The adapter restores the original training flag after successful inference. A custom `.predict()` method owns its own inference behavior.

For Torch portability, use an importable class or function with a constructor configuration dictionary (`talos_config`), or pass `model_factory=(factory, configuration)`. Generic containers such as `torch.nn.Sequential` need an explicit factory. Model and source checksum failures stop restoration; compatible installed framework versions are still required on the destination.

## Minimal adapter check

`talos.backends.backend_for(backend='torch').name` returns `'torch'` without fitting a model or importing Torch. This checks adapter dispatch only; the [native Torch walkthrough](Examples_PyTorch.md) verifies a complete training callback and restored predictions.

## Read next

Start with the [bounded Scan example](Scan.md#minimal-example), use the [native Torch walkthrough](Examples_PyTorch.md), or [migrate an existing model to an SFD](Migration.md).
