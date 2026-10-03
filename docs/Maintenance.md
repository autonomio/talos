# Maintenance and executable documentation

Maintain framework compatibility, reproducible records and archive/recovery behavior together. Update supported dependency floors when upstream fixes require it; keep the TensorFlow 2.14 compatibility lane explicit.

## Prerequisites and ownership

Use the supported Python/framework environments from
[Installation](Install_Options.md), a clean checkout and enough disk space for
trained artifacts and retained verification reports. The verifier creates
isolated local study projects, trains bounded real examples and builds an
unpublished wheel; its backup check pushes only to a local bare Git fixture.
Physical accelerator checks require separate available hardware.

Node.js 20.18.1 or later and locked npm dependencies are additionally required for
[site verification](Developer/Documentation-System.md). Python compatibility and
archive contracts belong to this page; rendering, routing and visual treatment
belong to the documentation system.

## Repeatable checks

Install the checkout with `test,plots,samplers,tensorflow,torch` extras, then run `python tools/verify_documentation.py --output-dir verification-output`. The command executes every product/example fenced block in README, CONTRIBUTING and the mapped documentation, all notebook code cells, both standalone scripts and all three native SFD examples. It rejects missing executions and changed source hashes. `--verify-only` rechecks retained receipts against the current sources.

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

The dependency audit gates core, current, minimum and legacy environments. The legacy lane builds hash-pinned owned Keras/Protobuf wheels, verifies the complete installed source tree and records each upstream advisory with its exact repair or absence proof. Unknown advisories, source drift, incomplete graphs and auditor errors fail. [Security backports](Developer/Security-Backports.md) owns patch maintenance and compatibility boundaries. Combining the legacy and modern extras is unsupported. Audit the exact resolved environment with `pip-audit --strict --format json --output audit.json`; retain its package versions and findings rather than treating successful installation as security evidence.

## Verification — 1 October 2026

Modern and supported minimum framework suites each pass 159 tests. The legacy source suite passes 149 with 10 optional Torch skips; installed core wheels pass 137 with 22 DL skips on both Python 3.10 and 3.13. Current and minimum dependency audits each cover 79 installed packages with no findings or skipped dependencies.

The pre-adoption source receipt at commit `c0db9eb` is saved in [verification/2026-10-01.json](verification/2026-10-01.json). It records execution scope, dependency audit results, framework versions and hashes of every documented block/example. CI repeats executable documentation and uploads its complete reports; the security workflow retains audit reports.

The [documentation adoption receipt](verification/2026-10-01-docs-adoption.json)
records content preservation and the later site/example acceptance against the
new route map. Its source hashes distinguish that candidate from the earlier
framework baseline.

Maintenance exposed and fixed two core defects: importing `talos.model` in a cold process caused a circular import, and paused Gamify edits were read after the next trial had already trained. Both have regression coverage. Historical template training defaults remain unchanged; educational examples declare smaller budgets.

Documentation updates retain the human-authored workflows while correcting obsolete framework APIs, disconnected model inputs, incomplete callback examples, classification encodings and machine-specific or remote notebook paths. Backend upgrades must preserve working callbacks, native artifact portability and historical archive readers together.

## Annual review and failure boundaries

1. Reassess upstream Python lifecycle and the supported framework matrix. Record
   a date, resolved versions and decisions about minimum and legacy lanes.
2. Audit resolved dependencies, update necessary floors and rerun the affected
   compatibility suites. Retain upstream findings and validate every owned backport; fail on unproved findings.
3. Exercise completed-trial resume, source/data mismatch rejection and
   fresh-process archive restoration, including caller-source deletion.
4. Execute the documented models, notebooks and CLI workflows; retain receipts
   tied to their exact source hashes. A missing or stale receipt fails acceptance.
5. Build and inspect the documentation with its locked dependencies, local search,
   source edit links, mobile layout and accessibility checks.
6. Build the distribution and install the wheel outside the checkout before
   recording a release baseline.

A CPU result cannot establish accelerator determinism or a physical power
reading. An old archive loading successfully cannot establish compatibility
with a future framework version. Keep those limits explicit in review evidence.

## Read next

[Contributing](../CONTRIBUTING.md) supplies development commands.
[Documentation system](Developer/Documentation-System.md) and
[Documentation style](Developer/Documentation-Style.md) cover site maintenance.
