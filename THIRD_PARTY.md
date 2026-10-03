# Third-party notices

Talos declares Python runtime and optional framework dependencies in `pyproject.toml`, CI toolchains in `requirements/ci/`, and the documentation toolchain in `docs-site/package.json` and its lockfile. Review dependency licenses when adding or materially upgrading packages.

## Source attribution

[NOTICE](NOTICE) retains required MIT attribution for copied infrastructure and documentation scaffolding. Talos owns its adopted implementation and evolves independently. Attribution identifies source ownership; it is not product branding.

The original Contributor Covenant attribution remains in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). The Autonomio documentation style source remains in `docs/_media/autonomio-style-guide.pdf`; Finlandica font packages carry their own license.

## Owned security backports

Keras 2.14.0 is Apache-2.0 software from the Keras authors; Protobuf 4.25.9
retains its Google BSD license. Their reconstructed wheels retain all original
license files, add the Keras Apache license absent from its upstream wheel, and add `AUTONOMIO_PATCHES.json` with official source hashes and
patch identities. Local versions identify Autonomio's changes separately from
upstream releases. Talos does not publish these projects to PyPI.

The documentation patches adapt MIT-licensed braces (Jon Schlinkert) and
BSD-2-Clause http-cache-semantics (Kornel Lesiński). `npm ci` retains their installed package
licenses; source distributions retain their complete notices in
`docs-site/security-patches/licenses/`. Patch manifests identify exact upstream
and repaired source bytes.
[Security backports](docs/Developer/Security-Backports.md) owns verification
and replacement with compatible upstream fixes.

## Vulnerability evidence

Python audits cover the core, current and supported minimum dependencies. The historical TensorFlow lane reports upstream advisories explicitly and requires exact owned-backport repair or absence proof. Time-limited exceptions require an advisory identifier, reason and expiry in `.github/vuln_exceptions.json`.

The documentation audit rejects advisories at every severity except its two explicitly approved, time-bounded entries, which also require verified installed security backports. Audit results change with advisory databases; use the reports for the exact candidate rather than treating an earlier clean run as a permanent assurance.

## Read next

- [Security policy](SECURITY.md)
- [Packaging](docs/Developer/Packaging.md)
