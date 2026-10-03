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

The protected assets are caller data and source, trained weights, experiment
records and their identities, provider credentials, and files reachable with
the local user's permissions. Relevant adversaries can alter a copied manifest
or archive, supply malformed declarative values, tamper with a network response,
or introduce a compromised dependency or CI contribution. Arbitrary model and
local strategy code remains trusted executable input; it runs with the caller's
permissions. Talos provides no isolation boundary for that code.

| Additional boundary | Control and evidence | Residual responsibility |
| --- | --- | --- |
| Remote entropy provider | `talos/reducers/remote_entropy.py` uses a default verified TLS context with TLS 1.2 minimum; rejects redirects and oversized or malformed responses; the sampler retains unique legal indexes. `tests/test_remote_entropy.py` exercises actual loopback certificate rejection before private headers or body, credential rotation and redirect refusal. | The caller selects a provider and supplies an authorized key; controlled tests do not prove live-provider service or physical entropy quality. |
| Provider key file | A caller-owned key file is read for each request; key rotation requires no code change. Keys are not embedded in model source or persisted by the provider client. | Restrict the file's permissions and keep keys out of source, logs and experiment exports. Trusted caller code can still read any file accessible to the process. |
| Declarative manifest and control files | Safe YAML parsing and field validation precede caller imports; committed content is rehashed before resolution, fork, recommit and reindex. `tests/test_manifest_validation.py` checks changed committed bytes and invalid objective/boolean controls before source hydration. | A valid SFD still names executable code. A matching digest establishes integrity against retained records, not an independent publisher's identity. |

These controls counter network impersonation, credential-bearing redirects,
malformed provider responses, path traversal and changed declarative records.
No private application data is transmitted before transport authentication in
the inspected key-bearing client path. Dataset HTTPS requests and Git backups
use their platform verification defaults. This assumes caller-owned operating
system trust stores and Git configuration retain certificate verification.
Network availability, certificate authorities, trusted dependencies and caller
credential hygiene remain external
assumptions. Legacy framework advisories remain unresolved as recorded below.

## Reviewed integration

The checked-in `master` snapshot requires green checks, eligible non-author approval, enforcement-surface code-owner review, resolved threads and an up-to-date branch, and blocks force pushes/deletion. Native CodeQL code-scanning protection blocks new security findings at every severity. A successful analysis/upload job alone does not establish an alert-free change. Copilot review requests run automatically; the ruleset separately requires an eligible non-author approval. The live ruleset gate detects drift; the privileged post-merge audit also inspects bypass actors.

Live ruleset `24306812` was active with exact snapshot parity and no bypass
actors when [PR 608](https://github.com/autonomio/talos/pull/608) merged on
1 October 2026. `bit-mis` supplied the eligible non-author approval. The
[post-merge privileged audit](https://github.com/autonomio/talos/actions/runs/36893895454)
passed with the scoped organization token, including bypass-actor inspection.
The [master CodeQL analysis](https://github.com/autonomio/talos/actions/runs/36893895563)
also passed. These observations apply to merge commit
`9783406eafd0c9d72d00010aeffb379a534a5349`; future settings and changes need
their own evidence.

## Secure design and common weaknesses

The controls below apply to local execution and repository integration. They
explain the implementation choices; they do not attest to a maintainer's
personal security knowledge or make hostile user code safe.

| Principle or weakness | Talos control and limit |
| --- | --- |
| Least privilege | PR workflows declare read permissions; release attachment has a separate write job without checkout or training execution |
| Complete mediation and fail-safe defaults | Issue parsers reject invalid contracts; release validation rejects source, signature and digest mismatches before attachment |
| Open design and economy of mechanism | Public source, documented interfaces and one executor serve Python, SFD and CLI callers |
| Separation of privilege | Protected integration requires an eligible non-author approval; PyPI publication has its own environment and enablement |
| Least common mechanism | Optional frameworks load through their declared backend; no shared hosted account or multitenant execution service exists |
| Psychological acceptability | Trust requirements are stated at archive/model entry points; invalid identities fail visibly |
| Limited attack surface and input validation | No finance service or hosted arbitrary-code endpoint; CLI/schema validation and containment checks guard recorded inputs |
| Defense in depth and unnecessary risk | Hash pins, dependency audits, CodeQL, reviewed integration and artifact verification cover different failure modes |
| Injection and untrusted deserialization | User callbacks, SFD Python and serialized framework objects require trusted origins; Talos does not advertise a sandbox |
| Path traversal and integrity | Archive extraction in `talos/commands/restore.py` validates destination containment; `talos/experiment/source_snapshot.py` checks source-bundle containment and recorded digests |
| Authentication and credential exposure | Talos has no account/password store; Actions credentials stay in declared jobs and are not persisted by checkout |

Scientific correctness is a separate boundary: causal preprocessing, metric
direction, seed semantics and recovery identity need behavioral tests even
when a security analyzer reports no findings.

## Law/configuration agreement

The honesty suite compares required contexts in `governance.yml`, the ten annotated workflow laws in CLAUDE and `.github/rulesets/master.json`. One server-side branch-protection law has no status context. Dropping any single mirrored description must fail the contract.

## Dependencies and credentials

Python dependency audits cover supported core/current/minimum lanes; legacy upstream advisories remain visible. Exceptions name an advisory, reason and expiry. Documentation rejects every advisory severity subject to [exact-version, expiring maintainer approvals](Documentation-System.md#documentation-dependency-exceptions); its original findings remain visible. Dependabot supplies update pressure; a green audit is evidence at its recorded time.

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
- Live branch protection and audit credentials have dated external proof; release execution, release environment and PyPI publisher still need verification.
- Continuity depends on eligible maintainers and reviewers in [MAINTAINERS.md](../../MAINTAINERS.md).

## Verify a claim

Run `python -m pytest governance/tests -q` for checked-in contracts and the documented library/framework/archive acceptance checks for behavior. For live protection, inspect `gh api repos/autonomio/talos/rulesets` with authorized read access. For a release built through the activated path, inspect its CI evidence and use `gh attestation verify ARTIFACT --repo autonomio/talos`.

Record candidate hashes and distinguish source, local execution, CI and live configuration evidence. A command returning zero is useful only when its assertion matches the claim.

## Read next

- [OpenSSF evidence](OpenSSF.md)
- [Release policy](Release-Policy.md)
- [Packaging](Packaging.md)
- [Configuration](Configuration.md)
