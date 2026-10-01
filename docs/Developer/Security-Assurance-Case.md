# Security assurance case

This page names Talos's trust boundaries and the evidence supporting each adopted control. It distinguishes local tooling from controls that require live activation. [SECURITY.md](../../SECURITY.md) owns private vulnerability reporting.

## Prerequisites

Inspect the exact candidate, governance test results, dependency audits, artifact checks and live settings when making a remote enforcement claim. [SETUP.md](../../SETUP.md) owns external activation.

## System and trust boundaries

Talos is a local scientific parameter-sweep library and CLI. It executes user-selected model code, writes experiment records and trained artifacts, and restores framework-native objects. It is not a hosted account service or code sandbox.

| Boundary | Input crossing it | Required assumption/control |
| --- | --- | --- |
| Callback/SFD execution | User Python and optional framework objects | Trusted source; explicit selection and visible failures |
| Archive restoration | Saved metadata and serialized model objects | Trusted origin; explicit factories/custom objects and fresh-process compatibility proof |
| Resume | Parameters, source/data identity and local state | Manifest consistency and mismatch rejection |
| Issue/PR parsing | Untrusted Markdown and GitHub JSON | Parser contract/property tests and fail-closed validation |
| CI supply chain | Dependencies, actions and tokens | Hash/SHA pins, vulnerability checks and least privilege |
| Publication | Approved tag and built artifacts | Explicit workflow, trusted publisher, digests and attestations |

A manifest establishes recorded identities and execution evidence. It does not certify arbitrary training code or sanitize a hostile archive.

## Reviewed integration

The checked-in `master` snapshot requires green checks, eligible non-author approval, enforcement-surface code-owner review, resolved threads and an up-to-date branch, and blocks force pushes/deletion. The live ruleset gate detects drift; the privileged post-merge audit also inspects bypass actors.

At adoption the complete check/review ruleset is a target control. Existing classic protection already covers administrators, force pushes and up-to-date branches, but supplies no required checks or reviews. Without an installed ruleset, `RULESET_ID` and the audit credential, local contract tests cannot prove reviewed integration is enforced on GitHub.

## Law/configuration agreement

The honesty suite compares required contexts in `governance.yml`, the ten annotated workflow laws in CLAUDE and `.github/rulesets/master.json`. One server-side branch-protection law has no status context. Dropping any single mirrored description must fail the contract.

## Dependencies and credentials

Python dependency audits cover supported core/current/minimum lanes; legacy upstream advisories remain visible. Exceptions name an advisory, reason and expiry. Documentation rejects known advisories at every severity. Dependabot supplies update pressure; a green audit is evidence at its recorded time.

Actions are pinned to full commits, toolchains to hashes, workflow tokens to declared permissions and checkout credentials disabled except where a declared release operation needs them. PR-controlled code must not execute in a privileged target event. Supply-chain contract tests establish these source properties; live secret scope still needs administrator verification.

## Artifacts and scientific recovery

The packaging plane proves fixed-epoch repeatability, distribution contents, bounded dependency metadata and installed-wheel behavior outside the checkout. The release path supports GitHub build provenance and SHA-256 digests with PyPI trusted publishing. These claims apply after activation and successful release execution; historical releases gain no provenance retroactively.

Training acceptance uses real bundled data, the three DL paths and the shared executor. Recovery proof runs in fresh processes, checks source/data identities and exercises source-independent restoration. These are correctness controls, not an assurance that untrusted models are safe to execute.

## Parser and static-analysis evidence

Property tests feed arbitrary strings to issue-body parsers and assert deterministic, non-crashing behavior. CodeQL, strict-quality debt, typing/fallback ratchets and coverage checks run on declared surfaces. Their budgets expose inherited limitations rather than claiming pristine source.

## Residual risks

- Model/SFD code and archive serialization require trusted origins.
- Legacy framework support can retain upstream advisories; current-lane success does not erase them.
- Static analysis and generated parser inputs cannot prove scientific validity or all runtime behavior.
- Live branch protection, credential scopes, release environment and publisher must be verified externally.
- Continuity depends on eligible maintainers and reviewers in [MAINTAINERS.md](../../MAINTAINERS.md).

## Verify a claim

Run `python -m pytest governance/tests -q` for checked-in contracts and the documented library/framework/archive acceptance checks for behavior. For live protection, inspect `gh api repos/autonomio/talos/rulesets` with authorized read access. For a release built through the activated path, inspect its CI evidence and use `gh attestation verify ARTIFACT --repo autonomio/talos`.

Record candidate hashes and distinguish source, local execution, CI and live configuration evidence. A command returning zero is useful only when its assertion matches the claim.

## Read next

- [Release policy](Release-Policy.md)
- [Packaging](Packaging.md)
- [Configuration](Configuration.md)
