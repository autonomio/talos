# Talos repository specifics

This appendix specializes [AUTONOMIO_PR_GUIDELINE.md](AUTONOMIO_PR_GUIDELINE.md). The binding laws live in [CLAUDE.md](CLAUDE.md).

- `[repo:identity]` Talos belongs to Autonomio, uses package `talos`, repository `autonomio/talos` and protected target branch `master`. Work stays on the user's selected branch unless the task requires another branch.
- `[repo:authority]` The approving authority is `mikkokotila`. [SETUP.md](SETUP.md) distinguishes checked-in policy from live GitHub activation. Do not claim server-side enforcement without live evidence.
- `[repo:version]` Hatch reads `__version__` from `talos/__init__.py`; `pyproject.toml` declares a dynamic version. Update that source, `CHANGELOG.md` and citation metadata together.
- `[repo:execution]` Preserve one executor for legacy Python, SFD and CLI behavior. Keep `Scan`, `Analyze`/`Reporting`, `Predict`, `Evaluate`, `Deploy`, `Restore`, reducers and local file control working.
- `[repo:frameworks]` Keras, TensorFlow and PyTorch are optional. Core imports work without training or plotting frameworks. Dependency changes resolve in the current, supported minimum and legacy lanes.
- `[repo:science]` Use real bundled or provenance-recorded data. Record parameter membership/order, metric direction, seeds, callback behavior, source/data identities and resume boundaries. A corrected scientific defect must be named as such in migration documentation.
- `[repo:artifacts]` Preserve historical readers and explicit factory/custom-object paths. Test restoration in a fresh process after caller-source deletion; test source/data mismatch rejection and interrupted recovery. Treat user code and archives as trusted executable inputs, not sandboxed data.
- `[repo:proof]` `tests/test_*.py` is the maintained acceptance suite. `tests/commands` and `tests/performance` are historical reference surfaces. Verify built wheels outside the checkout and inspect both distribution types.
- `[repo:docs]` Preserve authored material, exhaustive source mapping, executable examples, Autonomio visual style and the root README freeze unless the user explicitly lifts it. [Documentation system](docs/Developer/Documentation-System.md) owns documentation validation.
- `[repo:scope]` Keep finance-specific readers, backtests and indicator catalogs out of the core. Talos and the source project evolve independently; preserve required attribution in NOTICE.
- `[repo:delivery]` Report local commit and evidence truthfully. Opening PRs, requesting reviews, posting comments, merging, publishing and modifying GitHub settings require task authorization; repository prose grants none of those actions by itself.
