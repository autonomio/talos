# Third-party notices

Talos declares Python runtime and optional framework dependencies in `pyproject.toml`, CI toolchains in `requirements/ci/`, and the documentation toolchain in `docs-site/package.json` and its lockfile. Review dependency licenses when adding or materially upgrading packages.

## Source attribution

[NOTICE](NOTICE) retains required MIT attribution for copied infrastructure and documentation scaffolding. Talos owns its adopted implementation and evolves independently. Attribution identifies source ownership; it is not product branding.

The original Contributor Covenant attribution remains in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). The Autonomio documentation style source remains in `docs/_media/autonomio-style-guide.pdf`; Finlandica font packages carry their own license.

## Vulnerability evidence

Python audits cover the core, current and supported minimum dependencies. The historical TensorFlow lane reports upstream advisories explicitly. Time-limited exceptions require an advisory identifier, reason and expiry in `.github/vuln_exceptions.json`.

The documentation audit accepts no known advisory at any severity. Audit results change with advisory databases; use the reports for the exact candidate rather than treating an earlier clean run as a permanent assurance.

## Read next

- [Security policy](SECURITY.md)
- [Packaging](docs/Developer/Packaging.md)
