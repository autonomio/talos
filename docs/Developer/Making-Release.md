# Making a release

Talos releases automatically when a new reviewed package version merges to protected `master` and its full `Test and build` run succeeds. There is no additional release approval or enablement variable.

## Prerequisites

- A reviewed candidate integrated into protected `master`, with required CI checks and compatible distributions.
- A strictly advanced `talos/__init__.py` version, matching newest changelog section and citation metadata.
- A PyPI trusted publisher for `autonomio/talos`, workflow `deploy.yml`, environment `pypi`.
- The `pypi` environment permits protected branches without a required reviewer.

Read [Release policy](Release-Policy.md) for source, signature and burned-version contracts. External setup lives in [SETUP.md](../../SETUP.md).

## Automatic sequence

1. Prepare the version, changelog and citation metadata in the PR. Review core imports, installed wheels, affected framework lanes, documentation and archive/recovery evidence.
2. Merge after required checks and review pass. The normal protected-master CI verifies the merged source.
3. Its successful completion starts `deploy.yml`. The workflow validates the exact CI run and source, skips superseded heads, and derives `vMAJOR.MINOR.PATCH` from Hatch's version source.
4. The workflow creates the tag and GitHub release using the matching reviewed changelog. It downloads the exact Talos wheel and sdist tested by that CI run and audits them against the tagged source.
5. It reconstructs the two owned legacy security wheels, records `SHA256SUMS`, and signs all four distributions and the checksum file. It retains the authentic bundle as `talos-vMAJOR.MINOR.PATCH.sigstore.json`.
6. The attachment job verifies hashes, repository, signer workflow, source commit/ref and hosted runner before attaching the six immutable assets.
7. The separate protected PyPI job uploads only the Talos wheel and sdist automatically. Verify the registry version and an isolated installation, then retain the actual workflow, digests and consumer verification.

The source version is the release identity. Dependency-bot PRs must include the same version, changelog and citation bump as other PRs before merge. The required version gate enforces this; an existing tag or PyPI filename cannot be reused.

## Recovery dispatch

Use `gh workflow run deploy.yml --repo autonomio/talos --ref master -f tag=vMAJOR.MINOR.PATCH` only to recover the current master version. The command is a remote write. The selected source must have a successful protected-master full CI run, and the supplied tag must match its package version. The workflow enforces the same identity, signature and immutable-asset checks as automatic publication. A recovery dispatch needs no new routine release approval under the maintainer's standing merge-to-release authorization.

## Failure handling

Read the failed step and identify whether no release was created, a tag already exists, evidence failed, assets partly attached or PyPI publication partly succeeded. Existing GitHub assets cause attachment to fail rather than overwrite them. A retry before attachment must retain the original workflow/source identity; a later master commit cannot attest an older tag through manual dispatch. [Release policy](Release-Policy.md) defines asset and burned-version recovery. Do not change a version or bypass a failed test merely to make the upload proceed.

## Read next

- [Release policy](Release-Policy.md)
- [Packaging](Packaging.md)
