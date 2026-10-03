# OpenSSF evidence and maintenance

Talos has an [OpenSSF Best Practices application](https://www.bestpractices.dev/en/projects/15140)
and a [public Scorecard report](https://scorecard.dev/viewer/?uri=github.com/autonomio/talos).
Passing is awarded; the Silver application is 95% complete as of 3 October 2026. Silver is a Best Practices level;
Scorecard reports a separate score from zero to ten. This page records evidence
and outstanding work without claiming Silver attainment or future results.

## Prerequisites

Assess the exact public commit, recorded test environment and executed release.
A candidate workflow is evidence of its source contract; local tests cannot
establish that GitHub actually signed or published an artifact. Human criteria
require the named maintainer's confirmation.

## Published observations

| Observation | Scope and evidence |
| --- | --- |
| Scorecard 8.9/10 | Public API report dated 3 October 2026, 07:04:32 UTC; commit `19aba6539913d0e7cf0b9b52fefad8985dba58f9` |
| Best Practices application | Passing awarded at 100%; Silver 95% after the 3 October maintainer confirmation |
| Coverage and test policy | [PR 625](https://github.com/autonomio/talos/pull/625) merged at `06a635e`; its source tree equals the verified candidate `2388cec` |
| Reviewed integration | [PR 608](https://github.com/autonomio/talos/pull/608), approved by `bit-mis`, merged at `9783406`; PR 625 later merged at `06a635e` |
| Live protection | Active ruleset `24306812`; [privileged audit](https://github.com/autonomio/talos/actions/runs/36893895454) passed with exact snapshot parity and no bypass actors |
| Security analysis | [CodeQL on merged master](https://github.com/autonomio/talos/actions/runs/36893895563) passed; upload success alone does not certify absence of security defects |
| Published Scorecard workflow | [Master run](https://github.com/autonomio/talos/actions/runs/36893895393) succeeded with public results enabled |

The [Scorecard API](https://api.scorecard.dev/projects/github.com/autonomio/talos)
reports the commit and assessment date. Earlier scores of 3.1 and 7.7 refer to
earlier public source and assessment dates. Historic review and test activity continue to affect the
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
(77.56%), across 176 modules. The reviewed and merged 2.0.2 source measured 7,046 of 8,711
statements (80.89%) with 215 passing maintained tests and no skips, plus the
complete documentation corpus and examples; its 104 existing excluded lines
were unchanged. The [dated verification receipt](../verification/2026-10-01-openssf.json)
records local execution evidence. The complete coverage guard and all framework
lanes also passed in [hosted candidate CI](https://github.com/autonomio/talos/actions/runs/36914568173).
The [post-merge master run](https://github.com/autonomio/talos/actions/runs/36968443428)
also passed the framework matrix, executable corpus and complete coverage guard.
Its retained original report measures 7,054 of 8,711 statements (80.98%) over
all 176 modules; 215 maintained tests passed with no skips. This is distinct
from the earlier local measurement. New regression
cases exercise Pareto/diversity
selection, causal scaling and the packaged Keras, TensorFlow and PyTorch SFD
training and restoration paths. Use the selected source's executed report for its
result; do not substitute a projected total or narrow the measured package.

The [six-month fixed-bug ledger](../verification/2026-10-02-regression-ledger.json)
inventories the merged public source from 2 April through 2 October 2026.
Added regression assertions cover 16 of 25 independently identified fixed-defect
groups (64%). Nine uncredited groups remain in the denominator; feature additions,
open changes and unrepaired defects are excluded. The ledger binds every credited
assertion to immutable merged source and states the historical audit limits.

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

## Application review and remaining work

Use the official [passing](https://www.bestpractices.dev/en/criteria/0?details=true)
and [Silver criteria](https://www.bestpractices.dev/en/criteria/1?details=true).
Each answer must link to current public evidence or state a justified permitted
exception. Keep unsupported answers Unknown. In particular:

Mikko Kotila confirmed practical secure-design and common-vulnerability
knowledge. The maintainer also confirmed that zero-bang and bit-mis have the
capability, access and legal authority to continue development and issue a fix
within one week. The vulnerability-history attestation remains recorded in
the public application.

The two incomplete mandatory Silver criteria are dependency monitoring and
signed releases. The [owned backports](Security-Backports.md) require public
integration and verified CI; signed releases require an actual successful
release and consumer verification. Continue to:

- Review substantive static diagnostics. An inherited-debt ratchet is not a
  claim that every warning is fixed or every style exception is rare and local.
- Resolve legacy dependency advisories individually without treating trusted
  model execution as proof that a dependency vulnerability is unexploitable.
- Link an actual signed release and verify its artifacts from a consumer environment.
- Refresh the fixed-bug ledger after later merges; its current 64% result applies
  only to the recorded six-month public-source inventory.
- Reassess input, certificate and credential validation after the reviewed
  maintenance changes are integrated; candidate tests alone do not establish
  the deployed state.

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
