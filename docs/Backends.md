# Backends

Talos supports standalone Keras, TensorFlow/tf.keras and PyTorch. The core does not import a framework until model training, prediction, serialization or a framework-specific helper requires it.

## TensorFlow

Install `talos[tensorflow]` on Python 3.11 or newer (TensorFlow 2.20+, Keras 3.15+). Existing five-argument callbacks returning `(history, model)` work unchanged. The separate `legacy-tensorflow` lane preserves TensorFlow 2.14 / Keras 2.14 applications on Python 3.10–3.11. This compatibility lane retains known upstream advisories; modern extras are the default for new work.

## Keras

On Python 3.11 or newer, install `talos[keras,tensorflow]`, or `talos[keras,torch]` with `KERAS_BACKEND=torch`. Set the backend before importing Keras. Native `.keras` artifacts retain trained weights; pass `custom_objects` for custom layers/metrics when required.

## PyTorch

Install `talos[torch]` (Torch 2.13+) on Python 3.10 or newer. Return `(history, model)` or a structured SFD result. The documented historical return of a network carrying `.history` plus its parameter iterator is normalized to the trained network.

For portable restoration, provide an importable factory and constructor configuration, as demonstrated in [the Torch SFD](../examples/sfd/torch_sfd.py). Native artifacts use `state_dict`; nested class instances need an explicit restoration factory. Prediction preserves the network's existing training mode around inference.

See [migration](Migration.md) for archive and resume behavior.
