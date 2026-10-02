# Governance configuration

The repository's control surface has two canonical sources. `governance.yml` describes repository identity, scan paths, gate switches, review authority and policy settings. `.github/budgets.json` records measured ratchets. Workflows mirror settings that GitHub cannot read from the configuration; contract tests check the mirrors.

## Prerequisites

Read [CLAUDE.md](../../CLAUDE.md), install the hash-locked CI toolchains and inspect the current budget evidence. External protection setup belongs in [SETUP.md](../../SETUP.md).

## Repository shape

Talos uses package root `talos`, maintained tests under `tests`, governance tests under `governance/tests`, Python 3.12 for governance and `master` as the protected target. Historical `tests/commands` and `tests/performance` stay outside maintained pytest discovery. Runtime support remains Python 3.10–3.13.

The configuration declares the approving authority `mikkokotila`, the `slice` issue contract, Conventional Commits types, `# v{version}` changelog headers and the intended ruleset. The package version is dynamic: Hatch reads `talos/__init__.py`.

## Gate state

`enabled` controls whether a configurable gate runs. `required` controls whether its status context belongs in the intended required-check set. All adopted gates remain enabled. Every required context must occur once among the ten workflow laws in CLAUDE and in `.github/rulesets/master.json`; the honesty tests compare all three.

These declarations describe checked-in policy. Live enforcement exists only after an administrator installs it. A missing live ruleset or unreadable audit target fails its remote check rather than silently declaring success.

## Selected Python style

PEP 8 is the Python style guide. The existing required lint workflow separately
enforces the following selected style with zero findings, using the pinned Ruff
version and the strict configuration in `governance/ruff.toml`:

```sh
python -m ruff check --config governance/ruff.toml --preview --select E,W,I,D200,D205,D415,RUF022 --ignore E501 talos governance tests tools scripts examples
```

`E/W` checks pycodestyle conventions, `I` checks import ordering, `D200/D205/D415`
checks selected docstring conventions and `RUF022` checks `__all__` ordering.
Preview enables the selected whitespace checks. The existing `E501` exclusion
remains: the configured 100-character line length is not a hard limit in this
check. Legacy public API names remain compatibility contracts; naming rules
are not part of this selected command.

The existing per-file configuration leaves `D200/D205/D415` out of
`tests/**/*.py`, `governance/tests/**/*.py`, `governance/*.py` and `scripts/*.py`.
Whitespace and import ordering still apply there. Governance test fixtures
remain excluded by the existing repository configuration.

Any additional local style exception must be rare, justified in the source
at its location and reviewed. A measured debt allowance does not authorize a
style exception; the constitution still forbids new suppression comments.
The style check does not resolve annotation, complexity, dead-code or Pyright
findings. Those remain visible under their separate measured ratchets.

## Practical warning controls

The same required lint job compiles all six source scopes with `-Werror`;
compiler warnings fail before any training code is executed. Contract tests
place an invalid escape in each scope, verify rejection, then verify its raw
string correction. Runtime test warnings remain visible under pytest’s default
warning plugin, including deprecation and pending-deprecation warnings.

The existing strict Ruff selector and Pyright strict mode remain enabled.
Their diagnostics and error/warning ratchets expose inherited annotation,
complexity and optional-framework source-resolution limits; compiler-warning
cleanliness is not a claim that those separate inventories are empty.
No new warning suppression, selected-rule exclusion or debt allowance is added.

## Measured debt

Talos is an established scientific package. The initial strict-quality, typing, fallback, docstring, size, ratio, coverage and runtime measurements form the baseline. Existing debt remains visible in the budget file and reports; new work may not silently increase it. A basic Ruff invocation retains the existing low-level error check, while the strict profile in `governance/ruff.toml` is enforced through `check_quality_debt.py`.

Run `python governance/check_quality_debt.py` for strict Ruff and dead-code debt. The remaining scanner entry points live in `governance/` and are enumerated in the lint workflow. Read the exact failure before changing a budget.

| Budget | Allowed direction | Explicit relaxation |
| --- | --- | --- |
| Typing and fail-loud counts | Down | No widening in the judged PR |
| Strict quality and documentation debt | Down | No silent widening |
| Per-module line limits | Down | `[budget-raise: PATH: REASON]` |
| Coverage floors | Up | `[coverage-lower: FIELD: REASON]` |
| Runtime ceiling | Down | `[runtime-raise: REASON]` |

The ratchet compares the protected base's values and scan surfaces. Narrowing a scan path or excluding failing source cannot replace reducing debt.

## Initial migration boundary

The completed core predates this governance adoption. The one-time comparison
boundary is commit `94dd00ff16f9b57d79ce5cb6324d776aab495ced`, declared in
`adoption.baseline_commit`. The gate verifies that the protected base precedes
that commit, that the judged head descends from it and that it carries no
governance. Existing debt must match unchanged source from that boundary.
New changes receive the adopted checks.

Once the protected base contains governance, every comparison uses that base;
the migration boundary has no effect. This covers commit format, typing,
fail-loud patterns and changed-line coverage without rewriting earlier history.

## Absent and malformed values

A missing optional setting uses its documented default. A malformed setting blocks. Required identity and scan paths have no honest fallback: scanning an empty or renamed target must fail rather than pass. Repository-wide exclusions extend each scanner's exclusions; generated build trees stay out.

## Bot exemptions

Only the authors named in `automation.bot_authors` can skip the gates listed in `automation.exempt_gates`. Dependabot may skip slice and version requirements; the other gates still run. Empty lists remove the exemption. Every skip identifies the author and gate in output.

## Change a control

Change the reader, configuration, workflow mirror, law or snapshot and its tests together. A key with no consumer is dead configuration. Preserve exact required-check names. Record why a numerical relaxation is necessary and verify that it addresses the stated behavior rather than hiding a failing command.

## Read next

- [Technical debt](Technical-Debt.md)
- [Security assurance case](Security-Assurance-Case.md)
- [Developer home](README.md)

## CI runner allocation

`pr_checks_lint.yml` owns the separate required `pr_checks_tests` and
`pr_checks_lint` jobs. Product tests retain their runtime profile and ceiling;
governance contracts append coverage afterward. Lint consumes the successful
producer's immutable artifact ID from the same workflow run. Its receipt binds
the tested commit, run, attempt, lockfiles and coverage bytes; absent or changed
evidence fails the lint gate. Rerun the entire workflow to regenerate evidence
for a new attempt. The comment publisher has a separate write token and never
checks out or executes pull-request source.

Source checks cancel superseded heads and have total job timeouts. Core and
framework matrices each use at most two simultaneous runners. Installed-wheel
metadata, optional-backend imports, dependency consistency and acceptance run
in the existing core matrix. Strict supported dependency audits run inside the
already installed core, current and minimum lanes; legacy advisories remain
separately reported. Packaging still proves byte-identical builds and audits
both distribution types.

Title/body edits recheck Conventional Commits, slice acceptance, version and
budget, coverage and runtime waiver markers. They do not restart model tests
or documentation rendering. Dependabot groups each ecosystem's version and
security updates separately, limits open version PRs to one per ecosystem,
and staggers weekly version checks across Monday–Wednesday at 04:00 Helsinki
time. Security updates retain their immediate advisory-driven behavior.

All PR-body waiver decisions belong to the metadata-sensitive version job.
Source-only tests enforce measured runtime; lint enforces measured coverage,
quality and vulnerability checks. Adding or removing a waiver therefore clears
or fails the required version status without leaving stale body-dependent
failures on the required test or lint statuses.
