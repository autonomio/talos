# Packaging

The wheel contains installable Talos runtime files and package-level documentation. The source distribution contains the authored sources needed to inspect, test and build that wheel, including documentation and the adopted repository tooling.

## Prerequisites

Use the pinned build and packaging toolchains from `requirements/ci/`; `python -m build` invokes Hatch. Read the include/exclude contract in `pyproject.toml` and the artifact audit in `scripts/package_audit.py`.

## Artifact boundaries

| Artifact | Required content | Excluded content |
| --- | --- | --- |
| Wheel | `talos/`, bundled resources, package README and distribution metadata | Tests, repository governance and documentation frontend |
| Source distribution | Package, examples, maintained tests, authored docs, metadata and build/verification tooling | Installed dependencies, generated site, browser results, caches and local evidence trees |

Keep authored `docs-site` files available in the source distribution because documentation execution uses them. Exclude `node_modules`, `build`, `.generated`, `.docusaurus` and `test-results`. Generated reports belong in release evidence rather than the install artifact.

The audit checks required paths and forbidden prefixes. Metadata validity alone cannot prove distribution completeness. Every mapped documentation page must be present and byte-equal to its committed source in the source archive.

## Dynamic version

`pyproject.toml` declares `dynamic = ["version"]`; `[tool.hatch.version]` points to `talos/__init__.py`. The package's `__version__`, wheel metadata and source archive identity must agree. Do not introduce a second static version in project metadata.

## Dependencies

Runtime and optional dependencies have lower and upper bounds or an exact pin. Frameworks and plotting remain optional; importing core Talos cannot eagerly import TensorFlow, Keras, Torch, Matplotlib or PyArrow. CI toolchains are compiled and hash-locked separately from the runtime dependency envelope.

## Required proof

Build twice with a fixed `SOURCE_DATE_EPOCH`, compare artifact bytes and run the package audit. Check wheel/sdist metadata and declared dependency bounds. Install the wheel in a fresh environment outside the checkout, verify the distribution/package version, exercise core imports and CLI help, and run the maintained acceptance suite against installed code.

Backend changes additionally require the affected current, minimum or historical compatibility lanes and fresh-process restore after caller-source deletion. Packaging success cannot substitute for model behavior. Source archive checks also verify documentation inventory and authored frontend content.

## Read next

- [Contributing](../../CONTRIBUTING.md)
- [Release policy](Release-Policy.md)
- [Maintenance](../Maintenance.md)
