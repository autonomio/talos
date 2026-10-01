# Making a release

An authorized maintainer starts the release workflow explicitly. Merging to `master` does not release Talos.

## Prerequisites

- A reviewed candidate integrated into `master`, with required CI checks and compatible distributions.
- A strictly advanced `talos/__init__.py` version, matching newest changelog section and citation metadata.
- Explicit authority to create the tag/release.
- For PyPI publication, a configured trusted publisher, protected `pypi` environment and `PYPI_PUBLISH_ENABLED=true`.

Read [Release policy](Release-Policy.md) before acting. Verify the current local/live distinction in [SETUP.md](../../SETUP.md).

## Sequence

1. Review acceptance evidence for core imports, installed wheels, affected framework lanes, documentation and archive/recovery contracts.
2. Select the intended `master` candidate and tag `vMAJOR.MINOR.PATCH` matching Hatch's version. Verify the identity has never been released.
3. Dispatch `pr_post_release.yml` with that tag. Its script derives notes from the reviewed changelog, validates the tag and creates the GitHub release.
4. Inspect the release and exact commit. An existing tag is an idempotent creation case, not permission to reuse a published version.
5. Start `deploy.yml` explicitly for the same published tag when PyPI upload is authorized. A release created with `GITHUB_TOKEN` does not trigger another workflow from its release event. Publication verifies version availability, tests the released source, builds distributions, produces provenance and digests, then uses trusted publishing.
6. Confirm PyPI receives both artifacts and retain CI, digest, provenance, framework and documentation evidence. Add the actual release date to citation metadata as part of the reviewed release preparation.

The operator command is `gh workflow run pr_post_release.yml --repo autonomio/talos -f tag=vMAJOR.MINOR.PATCH`; substitute the approved concrete version. For the separate upload, use `gh workflow run deploy.yml --repo autonomio/talos -f tag=vMAJOR.MINOR.PATCH`. Both commands are remote writes, not local verification commands.

## Failure handling

Read the failed step and identify whether no release was created, a tag already exists, evidence failed or publication partly succeeded. [Release policy](Release-Policy.md) defines burned-version recovery. Do not change a version or bypass a failed test merely to make the upload proceed.

## Read next

- [Release policy](Release-Policy.md)
- [Packaging](Packaging.md)
