# Get help and report issues

Use the documentation to isolate a failure, then use the Talos issue tracker for a reproducible bug or feature request. This page identifies the current repository routes; older community services are not required to use Talos.

## Choose the route

| I want to… | Go to… |
|---|---|
| Troubleshoot an installation or callback | [Installation options](Install_Options.md), [Backends](Backends.md), and [Scan](Scan.md) |
| Troubleshoot resume or restoration | [SFD and CLI](SFD_and_CLI.md) and [Restore](Restore.md) |
| Report a reproducible bug | [Talos issues](https://github.com/autonomio/talos/issues) |
| Suggest a feature or discuss an implementation | [Talos issues](https://github.com/autonomio/talos/issues) and [contribution guidance](../CONTRIBUTING.md) |
| Find an existing community answer | [Talos issue archive](https://github.com/autonomio/talos/issues?q=is%3Aissue) |

The previous support page also pointed to a wiki and Spectrum chat. The maintained documentation and repository issue tracker are the canonical routes here.

## Prepare a useful report

No special tooling is required beyond the environment that reproduces the issue.

1. Confirm the problem against the relevant interface reference and [migration guide](Migration.md).
2. Reduce the model, input arrays and parameter dictionary to the smallest example that still reproduces it.
3. Record the Talos version, Python version, backend and backend version, operating system and command or Python entry point.
4. Include the complete exception and the expected versus observed outcome. For resume and archive issues, identify the artifacts involved and whether the source or configuration changed.
5. Open an issue with that reproduction. If the behavior depends on a physical device or external provider, state the device, driver or provider boundary explicitly.

A useful report lets a maintainer run the same failing path and determine whether the defect belongs to Talos, the callback or an optional service. A result metric alone does not establish a software defect; include the training setup and intended metric semantics.

Historical community discussions also appear on Stack Overflow. Search for
`autonomio talos` alongside the framework name.

## Read next

Use [maintenance verification](Maintenance.md) for compatibility evidence, or [contribution guidance](../CONTRIBUTING.md) to propose a fix with a focused regression check.
