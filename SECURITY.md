# Security policy

## Supported versions

The current maintained release line receives security fixes. Older framework compatibility lanes receive fixes where supported upstream versions allow them; unresolved upstream advisories must remain visible in release evidence.

## Report a vulnerability

Use [GitHub private security advisories](https://github.com/autonomio/talos/security/advisories/new). Include the affected version or commit, reproduction, impact and safe evidence. If private advisories are unavailable, contact <mailme@mikkokotila.com>. Do not post secrets, sensitive training data or an exploitable reproduction in a public issue.

Reporters receive credit in the fix's release notes unless they request otherwise.

## Trust boundaries

Model callbacks and SFD files are executable Python. Load only trusted code. A manifest records provenance; it does not sandbox a model. Archives may contain framework-native objects or Python serialization; restore only artifacts from trusted sources. See [Security assurance case](docs/Developer/Security-Assurance-Case.md).

## Release verification

The adopted publishing workflow supports GitHub build-provenance attestations, artifact SHA-256 digests and PyPI trusted publishing. These controls apply to releases produced after the workflow is activated; they are not claims about historical releases.

For an attested artifact, an operator can use `gh attestation verify ARTIFACT --repo autonomio/talos`. Check its digest against the publish run's job summary. Report a mismatch privately.

A CycloneDX SBOM and an offline provenance bundle are not produced. [Release policy](docs/Developer/Release-Policy.md) defines the artifact contract.
