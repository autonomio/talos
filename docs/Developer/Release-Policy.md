# Release policy

A Talos release publishes a reviewed version of the package with traceable artifacts. Governance adoption does not authorize a release. The operational sequence lives in [Making a release](Making-Release.md).

## Prerequisites

The candidate has passed the required local and CI checks, is reviewed against the affected framework and archive contracts, and has explicit release authorization. External setup follows [SETUP.md](../../SETUP.md).

## Version identity

Hatch reads `__version__` from `talos/__init__.py`; the newest `# vMAJOR.MINOR.PATCH` section in `CHANGELOG.md` must match. [Semantic versioning](Semantic-Versioning.md) chooses the bump. Update `CITATION.cff` with the version and the actual release date when released.

A tagged or uploaded version is burned. PyPI filenames cannot be reused after deletion, and deleted releases may disappear from its API. Never treat an absent JSON entry as permission to reuse an earlier published version.

## Release controls

| Control | Mechanism |
| --- | --- |
| Candidate selection | Explicit manual release workflow with a version tag matching the package |
| Reviewed integration | Required checks and reviews on the activated `master` ruleset |
| Release notes | Matching changelog section; no new prose at release time |
| PyPI enablement | `PYPI_PUBLISH_ENABLED` must be `true` |
| Upload authority | PyPI trusted publishing from the protected `pypi` environment |
| Existing filenames | Pre-build PyPI guard rejects already served versions |
| Artifact provenance | GitHub build-provenance attestations and publish-run SHA-256 digests |

The workflows are installed by this migration; the live controls and trusted publisher must be configured separately. Release creation with `GITHUB_TOKEN` needs explicit publication dispatch because its release event does not start another workflow. A normal merge does not create a release or publish the package.

## Deliverables

The wheel and source distribution are the install and inspection artifacts. The adopted publication workflow attests both and records their digests. Consumers of releases produced through that path can run `gh attestation verify ARTIFACT --repo autonomio/talos` and compare a local SHA-256 digest with the run summary.

No CycloneDX SBOM, offline provenance bundle or release-attached distribution assets are promised. Historical releases are not retroactively attested by adopting this workflow.

## Recovery

An existing tag makes release creation idempotent; it does not establish that publication completed. If a PyPI upload partly succeeds, advance the version and regenerate the reviewed changelog/citation identity. Never rerun a full upload expecting accepted filenames to be reusable. If only an attestation or evidence step failed, inspect its exact state before choosing the narrow repair.

## Read next

- [Making a release](Making-Release.md)
- [Packaging](Packaging.md)
- [Semantic versioning](Semantic-Versioning.md)
