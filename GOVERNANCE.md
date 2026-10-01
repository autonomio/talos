# Governance

Talos is an Autonomio project maintained by Mikko Kotila. Maintainers decide issue priority, product scope, release timing and review outcomes.

[CLAUDE.md](CLAUDE.md) defines the mechanical merge contracts. Changes to those contracts update the configuration, laws, tests and ruleset snapshot together. A maintainer decision cannot substitute for a passing required gate.

## Activation boundary

The repository contains the full governance tooling and the intended `master` ruleset. Live GitHub protection, required-check activation and privileged audit credentials require the administrator procedure in [SETUP.md](SETUP.md). A passing local contract test proves the checked-in policy, not that GitHub has installed it.

## Decision records

Record material decisions in an issue, pull request, release note or canonical documentation page. Include the affected public behavior, scientific consequence and acceptance evidence.

## Read next

- [Maintainers](MAINTAINERS.md)
- [Contributing](CONTRIBUTING.md)
- [Setup](SETUP.md)
