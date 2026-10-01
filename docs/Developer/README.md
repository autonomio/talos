# Developer documentation

Maintain working model callbacks, reproducible experiment records and portable
trained artifacts. This section covers contributor and maintainer tasks; the
public interfaces belong in [Reference](../Reference/README.md).

## Prerequisites

For library work, use a supported Python environment and the extras required by
the affected framework. Follow [CONTRIBUTING](../../CONTRIBUTING.md) for the
checkout and test commands. Documentation-site work additionally needs Node.js
20 or later and npm; it does not require a training framework.

## Choose the task

| Task | Canonical route | Expected proof |
| --- | --- | --- |
| Change library behavior | [Contributing](../../CONTRIBUTING.md) | Focused regression coverage, relevant framework lane and distribution checks |
| Update frameworks or archive contracts | [Maintenance](../Maintenance.md) | Current, minimum and legacy compatibility; fresh-process recovery |
| Author or restructure documentation | [Documentation system](Documentation-System.md) | Source-backed prose, exhaustive routes, runnable examples and full site checks |
| Change visual treatment | [Documentation style](Documentation-Style.md) | Measured desktop/mobile previews, keyboard access, light/dark contrast |
| Build and inspect the documentation site | [Site operation](../../docs-site/README.md) | Locked installation, audit, build, local search and browser proof |
| Participate in the project | [Code of conduct](../../CODE_OF_CONDUCT.md) | The stated community expectations |

## Review boundary

Document runtime behavior from the implementation and tests. A prose change
must not quietly change a model callback, manifest field, output default or
archive format. Example execution proves code behavior; the site build proves
rendering and route behavior. Keep those results separate in review evidence.

Before accepting a dependency update, run the affected framework and recovery
checks from [Maintenance](../Maintenance.md). Before accepting a documentation
change, run the commands in [Documentation system](Documentation-System.md).
A failure in either surface blocks its own acceptance; successful rendering
does not excuse a failed training example.

## Read next

Start with [Contributing](../../CONTRIBUTING.md) for a code change or
[Site operation](../../docs-site/README.md) for a documentation change.
