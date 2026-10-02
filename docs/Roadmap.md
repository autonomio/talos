# Development priorities

“Researchers first” guides Talos development: make automated deep learning workflows available to more researchers, and reduce the manual work required to use them. This page records that direction and the project’s established planning model; it is not a schedule of promised releases.

## Current boundary

Talos 2 preserves the Python Scan workflow and runs it through the shared SFD and CLI core. [Migration](Migration.md) describes that implemented boundary. [Capabilities](Overview.md) lists current user-facing features; planned ideas belong in the [issue tracker](https://github.com/autonomio/talos/issues) until implementation and verification establish them.

## Planning, testing and coding

The original roadmap allocated equal thirds to planning, testing and coding. Its internal priorities remain a useful review framework:

| Activity | Share within the activity | Purpose |
|---|---|---|
| Planning | One third | Design the future |
| Planning | One third | Write specifications |
| Planning | One third | Create documentation |
| Testing | One half | Hands-on use |
| Testing | One quarter | Add tests |
| Testing | One quarter | Improve existing tests |
| Coding | One third | Add features |
| Coding | One third | Improve current features |
| Coding | One third | Fix broken features |

These proportions express the project’s development approach rather than a measured current staffing allocation.

## Maintenance priorities

Annual compatibility and recovery maintenance verifies the declared Python/backend dependency matrix, all runnable documentation examples, held-out scientific metrics, archive restoration in fresh processes, and uninterrupted-versus-resumed trial equivalence. Refresh dependency/release metadata only after those checks pass. Physical GPU/power and external entropy services require separate environment-specific checks; CPU/provider fixtures cover the default workflow.

The executable procedure and accepted evidence belong in [maintenance verification](Maintenance.md). This page supplies direction rather than duplicating its commands or claiming tests that have not run.

## October 2026 to October 2027

This planning horizon runs from October 1, 2026 through October 1, 2027.
Maintainers review it when dependency support or scientific behavior changes.
Dates identify planned review periods, not promised releases.

| Period | Planned work | Acceptance evidence |
|---|---|---|
| October–December 2026 | Verify the adopted default-branch controls, full-framework coverage and the next authorized release's signatures | Required CI, complete-package statement coverage, scoped ruleset audit and verified release assets |
| January–March 2027 | Review Python, Keras, TensorFlow and PyTorch compatibility and upstream deprecations | Current/minimum resolver audits and framework training/archive tests |
| April–June 2027 | Review historical archive readers, custom objects and interrupted recovery | Fresh-process restoration, source/data mismatch rejection and resumed-run equivalence |
| July–September 2027 | Review metric direction, cohort selection, causal transforms and runnable documentation | Scientific behavior tests and the complete documentation execution manifest |
| By October 1, 2027 | Complete the annual maintenance procedure and revise this horizon | The retained maintenance verification record for the selected candidate |

Finance-specific indicators, backtesting, built-in data acquisition and hosted
model execution remain outside this plan. The focus is parameter sweeps,
scientific correctness, framework compatibility and reproducible recovery.

## Contribute

You can contribute by using Talos, writing or teaching about it, creating examples, recommending it, testing it, contributing code or regression checks, making feature requests, and improving the documentation.

To turn an idea into a reviewable change, first inspect the relevant current behavior, reproduce the proposed improvement with a bounded example, and follow [contribution guidance](../CONTRIBUTING.md). Implementation, documentation and meaningful proof establish when an idea becomes a current capability.

## Read next

Review [current capabilities](Overview.md), [maintenance verification](Maintenance.md), or [contribution guidance](../CONTRIBUTING.md). Use [support](Asking_Help.md) for a reproducible defect or feature request.
