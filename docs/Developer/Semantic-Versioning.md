# Semantic versioning

The version records compatibility of Talos's public Python, CLI, configuration, manifest and artifact contracts.

## Prerequisites

Inspect the affected public behavior, the current `talos/__init__.py` version and newest changelog section. Read [Release policy](Release-Policy.md) before choosing a release identity.

## Version surfaces

| Surface | Role |
| --- | --- |
| `talos/__init__.py` | Canonical `__version__`, read dynamically by Hatch |
| `pyproject.toml` | Dynamic version declaration and Hatch source path |
| `CHANGELOG.md` | Newest `# vMAJOR.MINOR.PATCH` section and release notes |
| `CITATION.cff` | Version and actual release date for scientific attribution |
| Git tag `vMAJOR.MINOR.PATCH` | Approved release identity |

## Choose the bump

- MAJOR: incompatible public API, CLI, schema, archive or package contract.
- MINOR: new compatible capability.
- PATCH: compatible correction, documentation, metadata or dependency refresh.

The PR's Conventional Commits type sets the minimum: `type!` requires a major bump, `feat` a minor bump and other types a patch bump. A larger justified bump is allowed. Human-authored PRs advance the version and changelog, including documentation work; explicit dependency-bot exemptions live in `governance.yml`.

Name a corrected scientific defect in the migration guide. Keeping wrong results for compatibility is not a release policy. Preserve historical artifact readers and state when an archive migration or explicit custom-object factory is required.

Never reuse an identity that has been tagged or uploaded. Citation metadata must not invent a release date for an unreleased candidate.

## Read next

- [Release policy](Release-Policy.md)
- [Making a release](Making-Release.md)
