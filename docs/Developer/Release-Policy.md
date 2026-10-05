# Release policy

Talos publishes each new reviewed package version automatically after its merge to protected `master` passes full CI. The maintainer has authorized this merge-to-release policy; publication needs no separate approval. The operational sequence lives in [Making a release](Making-Release.md).

## Prerequisites

The candidate has passed the required local and CI checks, is reviewed against the affected framework and archive contracts, and is integrated into protected `master`. External setup follows [SETUP.md](../../SETUP.md).

## Version identity

Hatch reads `__version__` from `talos/__init__.py`; the newest `# vMAJOR.MINOR.PATCH` section in `CHANGELOG.md` must match. [Semantic versioning](Semantic-Versioning.md) chooses the bump. Update `CITATION.cff` with the version and the actual release date when released.

A tagged or uploaded version is burned. PyPI filenames cannot be reused after deletion, and deleted releases may disappear from its API. Never treat an absent JSON entry as permission to reuse an earlier published version.

## Release controls

| Control | Mechanism |
| --- | --- |
| Candidate selection | Successful protected-master Test and build completion; derive the tag from the reviewed package version |
| Reviewed integration | Release commit belongs to protected `master`, with required checks and reviews |
| Source identity | Tag, checkout and workflow `GITHUB_SHA` identify the same commit |
| Release notes | Matching changelog section; no new prose at release time |
| GitHub artifacts | Exact Talos distributions from the selected full CI run, owned legacy wheels, signed checksums and authentic Sigstore bundle |
| Asset integrity | Verify hashes, signer workflow, source commit/ref and hosted runner before attachment |
| Existing assets | Reject matching filenames; never replace assets with `--clobber` |
| PyPI publication | Automatic after successful validation and signed GitHub attachment |
| Upload authority | PyPI trusted publishing from the protected `pypi` environment |
| Existing PyPI filenames | Pre-publication guard rejects already served versions |

`deploy.yml` starts on a successful `Test and build` completion for a push to protected `master`. It checks the run's repository, canonical workflow, event, branch, conclusion and source against GitHub's API. A superseded completion is skipped before tag creation. All release jobs share the same workflow source identity; there is no cross-workflow dispatch between tag creation and signing.

The workflow downloads the exact Python 3.12 distributions already built, audited and installed by that CI run. It audits their contents against the release checkout and checks their metadata again, without rerunning training or rebuilding Talos. The two owned legacy security wheels are reconstructed from their reviewed hashes. Signing, immutable GitHub attachment and PyPI upload retain separate permissions. The protected `pypi` environment allows protected branches and has no manual reviewer gate.

Every PR, including a dependency-bot PR, must advance the version and matching changelog/citation identity before merge. Bots retain only the human slice-issue exemption. A merge cannot reuse an existing tag or PyPI filename, and failed CI never publishes.

## Deliverables

After a successful `deploy.yml` run, the GitHub release receives the Talos wheel, source distribution, the two owned legacy security wheels, `SHA256SUMS` and `talos-vMAJOR.MINOR.PATCH.sigstore.json`, with the concrete release version in the bundle filename. The pinned attestation action signs all four distributions and the checksum file together. The bundle is copied directly from its `bundle-path` output and retains the authentic JSON Sigstore format.

The separate attachment job downloads the build artifacts without checking out or executing Talos. It checks the distribution hashes against `SHA256SUMS`, then verifies each distribution and the checksum file against the bundle before uploading any asset. Only the Talos wheel and source distribution enter the `release-dist` artifact used by PyPI.

The Sigstore bundle carries the signing certificate and its public verification
key. GitHub OIDC binds the short-lived certificate to the declared workflow
identity. The signing action creates an ephemeral private key on the hosted
runner, uses it for the attestation, and discards it; the artifact distribution
site receives no private signing key. Verification checks that identity and
the signed artifact digest rather than trusting a key downloaded without its
certificate chain.

The signed source artifact is the attached source distribution. GitHub
automatically generated source ZIP and tar archives are not subjects of this
attestation and must not be represented as signed distributions.

Consumers can use `gh attestation verify ARTIFACT --repo autonomio/talos`, or supply the downloaded bundle with `--bundle`. Match the signer workflow to `autonomio/talos/.github/workflows/deploy.yml`, the source digest/ref to the recorded release run, and require a GitHub-hosted runner. Verify the checksum file's attestation before checking its listed distribution hashes. A digest alone does not establish the producer's identity.

Source configuration and local contract tests do not establish release execution. Retain the successful public workflow run and independent verification of the actual release assets. Historical releases gain no provenance from this change. A CycloneDX SBOM is not produced.

## Recovery

An existing tag makes release creation idempotent; it does not establish that artifact attachment or PyPI publication completed. Inspect the original run and asset inventory before retrying. Signature, checksum or source-identity failures must be resolved before attachment.

The attachment job fails if any expected asset name already exists, including after a partial upload; it never deletes or overwrites accepted assets. Do not use a full rerun to replace them. If failure occurred before attachment, a retry must preserve the original reviewed source identity. A manual dispatch after `master` advances cannot substitute a different workflow commit for the old tag.

If a PyPI upload partly succeeds, advance the version and regenerate the reviewed changelog/citation identity. Never rerun a full upload expecting accepted filenames to be reusable. A recovery dispatch does not bypass existing GitHub assets. Any narrow recovery needs an explicitly reviewed procedure for the observed failure.

## Read next

- [Making a release](Making-Release.md)
- [Packaging](Packaging.md)
- [Semantic versioning](Semantic-Versioning.md)
