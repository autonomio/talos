# Technical debt

Record accepted limitations with their evidence and repair condition. Work not yet started belongs in an issue. The register below describes existing constraints; it does not claim they are harmless.

## Prerequisites

Inspect current implementation, measured budgets, acceptance evidence and the affected public documentation. Re-measure before changing a debt count.

## Current register

| ID | Surface and evidence | Current consequence | Repair trigger and path |
| --- | --- | --- | --- |
| TD-001 | Historical package quality/typing/docstring/fallback measurements in `.github/budgets.json` | Established source does not yet meet every strict rule at zero debt | Reduce debt while changing an affected module; preserve behavior and lower the measured baseline; update Configuration and relevant API docs |
| TD-002 | Legacy TensorFlow/Keras dependency lane and dependency audit reports | Upstream support/security differs from current framework lanes | Remove or revise support only through an explicit compatibility decision and migration route; update Maintenance, Security and framework requirements |
| TD-003 | Python callbacks, SFDs and framework-native archive readers | Executing/restoring untrusted sources can execute code; no sandbox is provided | A sandbox requirement needs a separately specified capability and threat model; update Security and archive documentation with any actual boundary change |
| TD-004 | GitHub activation state at repository migration | Checked-in policy does not enforce itself remotely | Administrator completes SETUP and records live evidence; remove this activation entry after verification |

Severity follows the concrete affected use path. Strict lint debt is a maintenance constraint; executable untrusted inputs are a trust boundary. Neither can be dismissed by passing an unrelated build.

## Register requirements

A new entry identifies the affected module/public surface, origin issue or measured evidence, current severity and blast radius, trigger for repair, migration/removal path and canonical docs to update. Use stable identifiers.

## Close an entry

Fix the affected implementation and acceptance checks, update canonical docs, and remove the active entry or replace it with a concise resolved note linking the evidence. Keep historical detail in Git history. Do not leave a stale mitigation claiming a repaired limitation remains.

## Read next

- [Configuration](Configuration.md)
- [Maintenance](../Maintenance.md)
- [Developer home](README.md)
