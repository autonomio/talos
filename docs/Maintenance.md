# Maintenance and executable documentation

Maintain framework compatibility, reproducible records and archive/recovery behavior together. Update supported dependency floors when upstream fixes require it; keep the TensorFlow 2.14 compatibility lane explicit.

## Repeatable checks

Install the checkout with `test,plots,samplers,tensorflow,torch` extras, then run `python tools/verify_documentation.py --output-dir verification-output`. The command executes every fenced block in README, CONTRIBUTING and the documentation, all notebook code cells, both standalone scripts and all three native SFD examples. It rejects missing executions and changed source hashes. `--verify-only` rechecks retained receipts against the current sources.

Examples train on real bundled Iris, breast cancer and digits observations. Documentation fragments use the declared page context or the complete held-out Iris setup from Scan. Native SFD examples train every declared combination and restore their archives in a fresh process after deleting the caller module. The guarded full Torch documentation example also restores after source deletion, with experiment execution, optimizer steps and data acquisition forbidden during restoration. There are no substitute training implementations. CLI checks use an isolated project, an unpublished local wheel and a local bare Git repository for backup.

NVIDIA command and power callback paths use an explicitly identified command provider. These checks verify integration behavior; physical GPU power readings, accelerator determinism and remote quantum entropy services require their own hardware/service checks.

Run the maintained acceptance suite, lint and distribution build described in [CONTRIBUTING](../CONTRIBUTING.md). Install the wheel outside the checkout. Core Python 3.10–3.13 must remain usable without DL or plotting frameworks. Test legacy, current and supported minimum frameworks; retain fresh-process native archive, deleted-helper recovery, interrupted-run, source/data verification and live-control regressions.

Python 3.10 reaches upstream end of life in October 2026; keep its tested Talos compatibility separate from upstream runtime maintenance. The next support review should reassess that lane and verify Python 3.14 before expanding the declared range. [Python lifecycle](https://devguide.python.org/versions/)

## Supported dependency floors

| Dependency | Minimum |
| --- | --- |
| TensorFlow | 2.20.0 |
| Keras | 3.15.0; Python 3.11+ |
| PyTorch | 2.13.0 |
| Protobuf with TensorFlow | 6.33.5 |
| scikit-learn | 1.5.0 |
| Click | 8.3.3 |
| tqdm | 4.66.3 |
| Requests | 2.33.0 |

The dependency audit gates core, current and minimum environments. TensorFlow 2.14 / Keras 2.14 remains a compatibility lane with upstream advisories, reported separately. Combining the legacy and modern extras is unsupported. Audit the exact resolved environment with `pip-audit --strict --format json --output audit.json`; retain its package versions and findings rather than treating successful installation as security evidence.

## Verification — 1 October 2026

Modern and supported minimum framework suites each pass 159 tests. The legacy source suite passes 149 with 10 optional Torch skips; installed core wheels pass 137 with 22 DL skips on both Python 3.10 and 3.13. Current and minimum dependency audits each cover 79 installed packages with no findings or skipped dependencies.

The source receipt is saved in [verification/2026-10-01.json](verification/2026-10-01.json). It records execution scope, dependency audit results, framework versions and hashes of every documented block/example. CI repeats executable documentation and uploads its complete reports; the security workflow retains audit reports.

Maintenance exposed and fixed two core defects: importing `talos.model` in a cold process caused a circular import, and paused Gamify edits were read after the next trial had already trained. Both have regression coverage. Historical template training defaults remain unchanged; educational examples declare smaller budgets.

Documentation updates retain the human-authored workflows while correcting obsolete framework APIs, disconnected model inputs, incomplete callback examples, classification encodings and machine-specific or remote notebook paths. Backend upgrades must preserve working callbacks, native artifact portability and historical archive readers together.
