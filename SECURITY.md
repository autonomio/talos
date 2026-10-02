# Security policy

## Supported versions

The current maintained release line receives security fixes. Older framework compatibility lanes receive fixes where supported upstream versions allow them; unresolved upstream advisories must remain visible in release evidence.

## Report a vulnerability

Use [GitHub private security advisories](https://github.com/autonomio/talos/security/advisories/new). Include the affected version or commit, reproduction, impact and safe evidence. If private advisories are unavailable, contact <mailme@mikkokotila.com>. Do not post secrets, sensitive training data or an exploitable reproduction in a public issue.

Reporters receive credit in the fix's release notes unless they request otherwise.

## Response and coordinated disclosure

The maintainer records reports privately in a GitHub security advisory or the
existing private email thread. Acknowledge a report within 14 days, identify an
owner, and agree on the next update date with the reporter. Never copy sensitive
data or an exploitable reproduction into a public issue.

Reproduce the finding, identify affected supported versions, assess severity,
and document whether trusted model execution or archive restoration is involved.
If the behavior lies outside Talos's trust boundary, explain that conclusion and
its evidence privately. Otherwise, prepare a fix or mitigation with a regression
test, run the affected acceptance checks, and obtain the required review.

Prioritize critical vulnerabilities immediately. Fix confirmed medium or higher
severity vulnerabilities within 60 days; if a dependency prevents a fix, record
the affected path, mitigation, expiry and update plan. A dependency exception is
not proof that a finding is unexploitable.

Coordinate publication with the reporter. Release the reviewed fix, name the
fixed version and impact in the release notes, request an advisory identifier
when appropriate, and credit the reporter unless they prefer anonymity. Publish
the advisory when the fix or mitigation is available. Close the private report
only after confirming the released artifact and notifying the reporter.

## Trust boundaries

Model callbacks and SFD files are executable Python. Load only trusted code. A manifest records provenance; it does not sandbox a model. Archives may contain framework-native objects or Python serialization; restore only artifacts from trusted sources. See [Security assurance case](docs/Developer/Security-Assurance-Case.md).

## Release verification

The release workflow builds signed GitHub assets independently of optional PyPI publication. It requires the selected tag, checkout and workflow commit to agree and the release commit to belong to protected master's history. PyPI still requires explicit enablement, trusted publishing and the protected `pypi` environment.

A successful release attaches the wheel, source distribution, signed `SHA256SUMS` and the authentic `talos-vMAJOR.MINOR.PATCH.sigstore.json` bundle. The separate attachment job verifies hashes, signer workflow, source commit/ref and hosted runner before uploading, and rejects existing asset names without replacement.

For an attested artifact, use `gh attestation verify ARTIFACT --repo autonomio/talos`, or provide the downloaded bundle with `--bundle`. Verify the checksum file's attestation and expected workflow/source identity before checking distribution hashes. [Release policy](docs/Developer/Release-Policy.md) defines the complete verification contract. Report a mismatch privately.

Actual signing and attachment require a successful authorized release run; local tests establish source controls only. Historical releases are not retroactively attested. A CycloneDX SBOM is not produced.
