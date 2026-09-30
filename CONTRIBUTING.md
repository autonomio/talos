# Contributing to Talos

Work on a branch and submit a reviewed pull request. Preserve working public Talos interfaces and use one executor for legacy Python, SFD and CLI behavior.

## Verification

```sh
pip install -e '.[test,plots,samplers,tensorflow,torch]'
ruff check talos tests/test_*.py
coverage run -m pytest -q
coverage report
python -m build
```

Use real bundled fixtures for training/inference checks. Test saved artifacts in a fresh process, and install the wheel outside the checkout. Core imports must succeed without DL or plotting frameworks. Add regression checks for meaningful behavior, including interruption recovery when changing persistence or queue control.

CI covers core Python 3.10–3.13, TensorFlow 2.14/NumPy 1.26, modern tf.keras and standalone Keras with a Torch backend. Backend dependencies remain optional; dependency changes must resolve in both legacy and modern lanes. Historical tests under `tests/commands` provide reference use patterns; maintained pytest contracts use current APIs and explicit real fixtures.

## Maintenance

At least annually, verify supported Python/framework versions, resolver compatibility, the complete acceptance suite, fresh-process archive round trips, source/data resume checks and dependency audit results. Check upstream deprecations and update tested versions/docs together. Backend evolution and persistence compatibility require more attention than the framework-independent sweep engine.

Update documentation with behavior changes. The migration guide identifies corrected defects rather than promising compatibility with wrong scientific output. Preserve legacy archive readers and explicit factory/custom-object paths.

Talos and Limen evolve independently. Generic infrastructure copied from Limen is owned here; preserve the MIT attribution in NOTICE. Keep finance, built-in experiment readers and indicator catalogs out of the Talos core. Do not add automatic data acquisition to the execution path.

Publishing requires a reviewed release and the configured PyPI trusted publisher/environment. The release workflow runs acceptance tests before building and publishing. No development task should merge or publish implicitly.
