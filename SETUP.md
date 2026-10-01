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

1. Review the adopted policy, all local evidence and the proposed snapshot in `.github/rulesets/master.json`.
2. Set `RULESET_AUDIT_TOKEN` as a read-only secret. Add the activation credential only when the administrator chooses to run activation.
3. Install the standard labels and apply the checked-in ruleset using the explicit `bootstrap_repository.yml` migration workflow, which runs `scripts/configure_repository.py --apply`. The migration must not rename the package, overwrite authored docs, reset the changelog or merge a seed PR.
4. The configuration script records the resulting ruleset identifier in `RULESET_ID`; verify it matches the installed ruleset.
5. Verify the live ruleset, required status-check contexts, non-author review, code-owner review, review-thread resolution, up-to-date branch requirement, force-push block and deletion block.
6. Open a reviewable validation PR only when authorized; confirm every required check runs and the ruleset gate compares the intended snapshot with live settings.
7. Confirm the privileged post-merge audit can read `bypass_actors`. A token that cannot see them cannot establish the complete protection contract.

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
