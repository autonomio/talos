# Making a release

An authorized maintainer starts the release workflow explicitly. Merging to `master` does not release Talos.

## Prerequisites

- A reviewed candidate integrated into protected `master`, with required CI checks and compatible distributions.
- For manual `deploy.yml` dispatch, the release tag matches the selected master workflow commit; keep this identity while preparing the release.
- A strictly advanced `talos/__init__.py` version, matching newest changelog section and citation metadata.
- Explicit authority to create the tag/release.
- For PyPI publication, a configured trusted publisher, protected `pypi` environment and `PYPI_PUBLISH_ENABLED=true`.

Read [Release policy](Release-Policy.md) before acting. Verify the current local/live distinction in [SETUP.md](../../SETUP.md).

## Sequence

1. Review acceptance evidence for core imports, installed wheels, affected framework lanes, documentation and archive/recovery contracts.
2. Select the intended `master` candidate and tag `vMAJOR.MINOR.PATCH` matching Hatch's version. Verify the identity has never been released.
3. Dispatch `pr_post_release.yml` with that tag. Its script derives notes from the reviewed changelog, validates the tag and creates the GitHub release.
4. Inspect the release and exact commit. An existing tag is an idempotent creation case, not permission to reuse a published version.
5. Start `deploy.yml` explicitly on `master` for the same published tag. A release created with `GITHUB_TOKEN` does not trigger another workflow from its release event. The tag, checkout and workflow commit must agree, and the tag must belong to protected master's history; stop if the selected source identity has changed.
6. Inspect the build and attachment jobs. The workflow tests the released source, builds the wheel and sdist, records `SHA256SUMS`, and signs all three subjects. It retains the attestation action's authentic bundle as `talos-vMAJOR.MINOR.PATCH.sigstore.json`, verifies hashes and signatures, and attaches the four release assets without replacing existing names. This runs independently of PyPI enablement.
7. Confirm the attached artifacts and checksum file verify against the bundle, expected signer workflow, source commit/ref and hosted runner. Retain the actual run, signatures, digests, framework and documentation evidence. Until this succeeds on a real release, the signed-release path has no operational proof.
8. When PyPI publication is authorized and enabled, inspect the separate protected `pypi` job after build and attachment succeed. Confirm PyPI receives only the wheel and sdist. Add the actual release date to citation metadata as part of the reviewed release preparation.

The operator command is `gh workflow run pr_post_release.yml --repo autonomio/talos --ref master -f tag=vMAJOR.MINOR.PATCH`; substitute the approved concrete version. For signed GitHub artifacts and optional PyPI publication, use `gh workflow run deploy.yml --repo autonomio/talos --ref master -f tag=vMAJOR.MINOR.PATCH`. Both commands are remote writes, not local verification commands. The PyPI variable and trusted publisher must be configured before dispatch when PyPI publication is intended.

## Failure handling

Read the failed step and identify whether no release was created, a tag already exists, evidence failed, assets partly attached or PyPI publication partly succeeded. Existing GitHub assets cause attachment to fail rather than overwrite them. A retry before attachment must retain the original workflow/source identity; a later master commit cannot attest an older tag through manual dispatch. [Release policy](Release-Policy.md) defines asset and burned-version recovery. Do not change a version or bypass a failed test merely to make the upload proceed.

## Read next

- [Release policy](Release-Policy.md)
- [Packaging](Packaging.md)
