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
| Candidate selection | Explicit release creation with a version tag matching the package |
| Reviewed integration | Release commit belongs to protected `master`, with required checks and reviews |
| Source identity | Tag, checkout and workflow `GITHUB_SHA` identify the same commit |
| Release notes | Matching changelog section; no new prose at release time |
| GitHub artifacts | Tested distributions, signed checksums and authentic Sigstore bundle; independent of PyPI enablement |
| Asset integrity | Verify hashes, signer workflow, source commit/ref and hosted runner before attachment |
| Existing assets | Reject matching filenames; never replace assets with `--clobber` |
| PyPI enablement | `PYPI_PUBLISH_ENABLED=true` and successful build and GitHub attachment |
| Upload authority | PyPI trusted publishing from the protected `pypi` environment |
| Existing PyPI filenames | Pre-build guard rejects already served versions when PyPI publication is enabled |

A published release event can start `deploy.yml`. Release creation with `GITHUB_TOKEN` needs explicit dispatch because its release event does not start another workflow. For manual dispatch, select `master`: the tag must match that workflow run's master commit. The workflow also fetches protected `master` and rejects a tag outside its history. These checks prevent a current workflow from attributing a historical checkout to a different attested source.

The build and GitHub asset attachment run without `PYPI_PUBLISH_ENABLED`. PyPI publication remains a separate gated job and requires a configured trusted publisher and protected environment. A normal merge does not create a release or publish the package.

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

Actual GitHub OIDC signing and release attachment remain unproven until an authorized Talos release completes this path and its assets verify. Source configuration and local contract tests do not establish release execution. Historical releases gain no provenance from this change. A CycloneDX SBOM is not produced.

## Recovery

An existing tag makes release creation idempotent; it does not establish that artifact attachment or PyPI publication completed. Inspect the original run and asset inventory before retrying. Signature, checksum or source-identity failures must be resolved before attachment.

The attachment job fails if any expected asset name already exists, including after a partial upload; it never deletes or overwrites accepted assets. Do not use a full rerun to replace them. If failure occurred before attachment, a retry must preserve the original reviewed source identity. A manual dispatch after `master` advances cannot substitute a different workflow commit for the old tag.

If a PyPI upload partly succeeds, advance the version and regenerate the reviewed changelog/citation identity. Never rerun a full upload expecting accepted filenames to be reusable. Enabling PyPI later does not start publication or bypass existing GitHub assets. Any narrow recovery needs an explicitly reviewed procedure for the observed failure.

## Read next

- [Making a release](Making-Release.md)
- [Packaging](Packaging.md)
- [Semantic versioning](Semantic-Versioning.md)
