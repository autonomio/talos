# Contributing to Talos

Read [CLAUDE.md](CLAUDE.md) and [Talos repository specifics](TALOS_REPO_SPECIFICS.md). Work on a branch and submit a reviewed pull request when the task authorizes one. Preserve working public Talos interfaces and use one executor for legacy Python, SFD and CLI behavior.

## Verification

```sh
pip install -e '.[test,plots,samplers,tensorflow,torch]'
ruff check talos tools tests/test_*.py
coverage run -m pytest -q
coverage report
python -m build
```

Use real bundled fixtures for training/inference checks. Test saved artifacts in a fresh process, and install the wheel outside the checkout. Core imports must succeed without DL or plotting frameworks. Add regression checks for meaningful behavior, including interruption recovery when changing persistence or queue control.

Run `python tools/verify_documentation.py --output-dir verification-output` to execute every documentation fence, all notebook cells, standalone scripts and the three SFD examples. The generated manifest rejects missing, failed or stale source hashes. Training uses real bundled data; hardware provider checks are identified explicitly. Retain the reports with the release evidence.

CI covers core Python 3.10–3.13, TensorFlow 2.14/NumPy 1.26, modern tf.keras and standalone Keras with a Torch backend. Backend dependencies remain optional. The supported minimum lane checks TensorFlow 2.20, Keras 3.15, Torch 2.13 and Protobuf 6.33.5; the dependency audit gates current/core/minimum lanes and reports legacy upstream advisories separately. Dependency changes must resolve in both legacy and modern lanes. Historical tests under `tests/commands` provide reference use patterns; maintained pytest contracts use current APIs and explicit real fixtures.

## Maintenance

At least annually, verify supported Python/framework versions, resolver compatibility, the complete acceptance suite, fresh-process archive round trips, source/data resume checks and dependency audit results. Check upstream deprecations and update tested versions/docs together. Backend evolution and persistence compatibility require more attention than the framework-independent sweep engine.

Update documentation with behavior changes. The migration guide identifies corrected defects rather than promising compatibility with wrong scientific output. Preserve legacy archive readers and explicit factory/custom-object paths.

Talos and Limen evolve independently. Generic infrastructure copied from Limen is owned here; preserve the MIT attribution in NOTICE. Keep finance, built-in experiment readers and indicator catalogs out of the Talos core. Do not add automatic data acquisition to the execution path.

Publishing requires a reviewed release and the configured PyPI trusted publisher/environment. The release workflow runs acceptance tests before building and publishing. No development task should merge or publish implicitly.

## Governance checks

Install the pinned toolchains in `requirements/ci/` using their hash-locked files. For the complete local contributor environment, use `pip install -e '.[dev,plots,samplers,tensorflow,torch]'`. Run `python governance/check_quality_debt.py` for the strict quality ratchet; the basic Ruff command above remains a compatibility check. Run `python -m pytest governance/tests -q` for repository contracts. The lint workflow enumerates the current scanners, measured debt ratchets, coverage and runtime checks; [Configuration](docs/Developer/Configuration.md) explains their settings. Test authoring may generate parser inputs; training evidence uses the real fixtures specified above.

A normal human-authored PR closes one open slice issue, matches its title and declared surfaces, advances the Hatch version in `talos/__init__.py` and adds a matching changelog section. The parent PRD closes with the slice only when it is the final open sub-issue. Dependency-bot exemptions are explicit in `governance.yml`.

Review against [.github/copilot-instructions.md](.github/copilot-instructions.md). Separate local test results from live CI and branch-protection status. [SETUP.md](SETUP.md) records activation of external controls; [Packaging](docs/Developer/Packaging.md) and [Release policy](docs/Developer/Release-Policy.md) define distribution and publication evidence.
