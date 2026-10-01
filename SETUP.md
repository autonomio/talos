# Repository activation

Talos is an existing repository adopting governance tooling. Preserve its package, history, version source, examples and documentation. This runbook configures the external services; it does not recreate or reseed the repository.

## Current boundary

The checked-in workflows, gate code and intended `master` ruleset are locally testable. At adoption, live Talos has no installed ruleset and no `RULESET_ID` variable. Existing classic protection covers administrators, force pushes and up-to-date branches; it has no required checks or reviews. Preserve those existing protections while activating the adopted policy. Local success is not live enforcement. An administrator must complete the steps below before relying on required checks, review protection or the post-merge audit.

## Prerequisites

- Administration access to `autonomio/talos` and access to Actions settings.
- GitHub Actions and CodeQL availability for this public repository.
- An eligible non-author code owner with write access for governed surfaces. `governance.yml` and `CODEOWNERS` currently name only `mikkokotila`. Before enabling code-owner enforcement for maintainer-authored PRs, designate another eligible owner in a reviewed change; adding a collaborator alone does not satisfy code-owner approval.
- Copilot code review availability for the intended ruleset.
- A protected `repository-administration` environment for activation and a protected `pypi` environment for release/publication.
- A read-only ruleset audit token and, only for the activation operation, a separately scoped administrator credential.

## Credentials and variables

| Name | Use | Required permissions |
| --- | --- | --- |
| `REPO_BOOTSTRAP_TOKEN` | Explicit existing-repository activation | Contents, Pull requests, Issues, Administration, Variables and Workflows read/write; Metadata read |
| `RULESET_AUDIT_TOKEN` | Post-merge live ruleset audit, including bypass actors | Administration and Metadata read |
| `RULESET_ID` | Identify the installed ruleset | Set after the ruleset is created |
| `PYPI_PUBLISH_ENABLED` | Enable the separate PyPI workflow | Leave unset until trusted publishing and the `pypi` environment are configured |

Use repository secrets or organization secrets restricted to Talos. Never print token values or store them in source. `GITHUB_TOKEN` cannot administer rulesets and events it creates generally do not start another workflow run; it cannot replace an activation credential.

`.github/labels.json` supplies the standard labels locally. Activation creates or updates the declared labels and preserves unrelated pre-existing labels; it does not depend on another organization's repository.

## Activate existing-repository governance

1. Review the adopted policy, all local evidence and the proposed snapshot in `.github/rulesets/master.json`. Reconcile the standard labels without deleting existing labels.
2. Choose another eligible code owner for maintainer-authored PRs and include that owner in the reviewed integration change. Protect the `repository-administration` environment and use read-only default workflow permissions.
3. Provision `RULESET_AUDIT_TOKEN` for Talos only. Verify its actual response includes `bypass_actors`; the permission label alone cannot prove complete observability. Keep the existing broad CLI OAuth credential out of Actions.
4. Open and validate the adoption PR through the existing protection, then integrate the reviewed workflows. The initial ruleset check cannot pass while its live prerequisite is absent; record that bootstrap boundary explicitly. Do not install mandatory check contexts before their workflows are available and validated.
5. Wait for the integrated master CodeQL analysis. Analysis/upload success is separate from the native `code_scanning` rule, which blocks new security findings at every severity. Copilot's rule requests reviews automatically; it does not itself require completion or an approving verdict.
6. Apply the reviewed labels and ruleset with `scripts/configure_repository.py --apply` using the authorized local administrator credential, or dispatch the protected `bootstrap_repository.yml` workflow from master after separately provisioning its activation credential. Neither path renames Talos, rewrites authored docs or resets history. The script records the installed identifier in `RULESET_ID`.
7. Verify exact live snapshot parity, including bypass actors, all required checks, CodeQL security thresholds, non-author/code-owner approval, resolved threads, up-to-date branches and force-push/deletion blocks. Preserve existing classic protection.
8. Dispatch the integrated privileged audit and confirm the scoped credential can read the complete live ruleset. Validate enforcement on a subsequent PR; do not declare activation complete from source tests or an API write alone.

For read-only verification, use `gh variable list --repo autonomio/talos`, `gh api repos/autonomio/talos/rulesets` and `gh label list --repo autonomio/talos`. The configured ruleset name and gate list come from `governance.yml`; compare the responses rather than merely checking that an API call succeeded.

## Enable releases separately

[Making a release](docs/Developer/Making-Release.md) defines the controlled release sequence. Do not enable automatic publication as a side effect of governance adoption.

Configure the PyPI project's trusted publisher for `autonomio/talos`, the publish workflow and the `pypi` environment. Protect that environment with the intended release approval. Verify the package name, version source and burned-version guard, then set `PYPI_PUBLISH_ENABLED=true` only when publication is authorized. Legacy PyPI tokens are not needed by the trusted-publishing path.

## Failure modes

| Symptom | Cause | Resolution |
| --- | --- | --- |
| No required checks on a PR | Workflows have not reached the target branch or the live ruleset is absent | Integrate through review, then install and verify the intended policy |
| Ruleset gate cannot read a target | `RULESET_ID` missing, wrong or unreadable | Set the actual identifier and verify repository/credential access |
| Audit cannot inspect bypass actors | Audit token lacks Administration read | Repair the read-only token scope; do not suppress the check |
| Review does not satisfy protection | Reviewer lacks write access, authored the PR or code-owner approval is missing | Obtain an eligible non-author review |
| Copilot review unavailable | Account or repository capability not configured | Configure it before declaring the intended protection active |
| Publication is skipped | Enablement variable or release prerequisite absent | Complete the separate release setup only when intended |
| PyPI guard rejects a version | That version has already been served | Advance the version; never reuse a burned identity |

## Read next

- [Configuration](docs/Developer/Configuration.md)
- [Security assurance case](docs/Developer/Security-Assurance-Case.md)
- [Making a release](docs/Developer/Making-Release.md)
