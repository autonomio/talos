# OpenSSF evidence and maintenance

Talos has an [OpenSSF Best Practices application](https://www.bestpractices.dev/en/projects/15140)
and a [public Scorecard report](https://scorecard.dev/viewer/?uri=github.com/autonomio/talos).
The Best Practices application is in progress. Silver is a Best Practices level;
Scorecard reports a separate score from zero to ten. This page records evidence
and outstanding work without claiming either badge attainment or future results.

## Prerequisites

Assess the exact public commit, recorded test environment and executed release.
A candidate workflow is evidence of its source contract; local tests cannot
establish that GitHub actually signed or published an artifact. Human criteria
require the named maintainer's confirmation.

## Published observations

| Observation | Scope and evidence |
| --- | --- |
| Scorecard 7.7/10 | Public API report dated 1 October 2026, 16:42:55 UTC; commit `9783406eafd0c9d72d00010aeffb379a534a5349`; Scorecard v5.3.0 |
| Reviewed integration | [PR 608](https://github.com/autonomio/talos/pull/608), approved by `bit-mis`, merged at the same commit |
| Live protection | Active ruleset `24306812`; [privileged audit](https://github.com/autonomio/talos/actions/runs/36893895454) passed with exact snapshot parity and no bypass actors |
| Security analysis | [CodeQL on merged master](https://github.com/autonomio/talos/actions/runs/36893895563) passed; upload success alone does not certify absence of security defects |
| Published Scorecard workflow | [Master run](https://github.com/autonomio/talos/actions/runs/36893895393) succeeded with public results enabled |

The [Scorecard API](https://api.scorecard.dev/projects/github.com/autonomio/talos)
reports the commit and assessment date. An earlier score of 3.1 referred to the
pre-adoption source. Historic review and test activity continue to affect the
new score; a workflow edit cannot change past reviewed changesets.

## Coverage and regression proof

Silver requires at least 80% statement coverage. Measure all `talos` Python
modules with the maintained acceptance suite, executable documentation and
examples in the current full-framework environment. Optional-framework skips
in the core-only lane are not proof of framework behavior.

`.github/coverage-full.toml` instruments subprocesses and retains the original
package as the report scope. `tools/check_statement_coverage.py` rejects
missing modules, duplicate or foreign checkout paths, inconsistent totals and
coverage below 80%, using statement counts rather than the combined line/branch
percentage. The documentation job in `.github/workflows/ci.yml` runs the check
and retains its JSON report. The existing required lint gate keeps its separate
core coverage floor and ratchet.

The pre-improvement full-framework measurement was 6,756 of 8,711 statements
(77.56%), across 176 modules. The 2.0.2 candidate measured 7,046 of 8,711
statements (80.89%) with 215 passing maintained tests and no skips, plus the
complete documentation corpus and examples; its 104 existing excluded lines
were unchanged. The [dated verification receipt](../verification/2026-10-01-openssf.json)
records local execution evidence pending public integration. New regression
cases exercise Pareto/diversity
selection, causal scaling and the packaged Keras, TensorFlow and PyTorch SFD
training and restoration paths. Use the candidate's executed report for its
result; do not substitute a projected total or narrow the measured package.

## Release evidence

[Release policy](Release-Policy.md) owns the signed-artifact contract and
[Making a release](Making-Release.md) owns the authorized sequence. Signature
verification must bind the artifact, repository, signer workflow, source commit
and source ref. An authentic Sigstore bundle and attested SHA-256 checksum list
belong with each new release's wheel and source distribution. Historical
releases remain unsigned unless their actual evidence says otherwise.

A successful reviewed release execution is still needed for the signed-release
criterion. PyPI enablement is a separate decision; the presence of organization
credentials does not establish a trusted publisher or release approval.

## Application review and remaining facts

Use the official [passing](https://www.bestpractices.dev/en/criteria/0?details=true)
and [Silver criteria](https://www.bestpractices.dev/en/criteria/1?details=true).
Each answer must link to current public evidence or state a justified permitted
exception. Keep unsupported answers Unknown. In particular:

- Name a primary developer who confirms secure-design and common-vulnerability
  knowledge; policies and generated prose cannot establish that expertise.
- Confirm six-month private vulnerability-response history, twelve-month
  reporter credit, and disposition of any publicly known medium-or-higher
  vulnerability older than 60 days.
- Name an independent backup maintainer and confirm their ability, credentials
  and legal authority to continue development and issue a fix within one week.
- Review substantive static diagnostics. An inherited-debt ratchet is not a
  claim that every warning is fixed or every style exception is rare and local.
- Link actual signed-release and full-package coverage evidence when available.

The [security assurance case](Security-Assurance-Case.md) records trust
boundaries; [CONTRIBUTING](../../CONTRIBUTING.md) records contribution and test
policy; [SECURITY.md](../../SECURITY.md) records vulnerability handling; the
[roadmap](../Roadmap.md) covers the next year. Their publication establishes
policy, not retrospective compliance.

The portal requires application data to be submitted under the Community Data
License Agreement–Permissive Version 2.0. The authorized project representative
must accept those terms before submission. After a level is attained, publish
its achievement link on the repository front page or live project website
within 48 hours, as the criteria require. The README links the live OpenSSF application and measured coverage publication;
the maintainer explicitly lifted its prior freeze for this update. An undeployed
documentation site does not establish achievement notice.

## Annual review

Re-run the current and minimum framework lanes, executable documentation,
full-package coverage, dependency audits, recovery and installed-wheel proof.
Inspect live ruleset parity, reviewer access, backup authority and publisher
configuration. Check public Scorecard and badge answers against the current
commit; refresh expired exceptions and release evidence. Record the date and
commit for each claim rather than carrying forward an old green status.

## Read next

- [Maintenance](../Maintenance.md)
- [Security assurance case](Security-Assurance-Case.md)
- [Release policy](Release-Policy.md)
