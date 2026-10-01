# Autonomio PR Guideline

Status: canonical universal PR guideline for Autonomio agent work.
Scope: rules that apply across Autonomio repositories unless a local repository entrypoint narrows them.
Source basis: adopted repository-template engineering discipline, specialized for Autonomio with user authorization taking precedence.
Specifics: repository-only details live in [TALOS_REPO_SPECIFICS.md](TALOS_REPO_SPECIFICS.md).

## Rubric

| Tag family | Meaning |
| --- | --- |
| `authority:*` | instruction precedence, routed reads, escalation |
| `scope:*` | task shape, slice boundaries, changed surfaces |
| `github:*` | branches, commits, issues, PRs, reviews, CI, merge |
| `change:*` | implementation discipline and engineering constraints |
| `evidence:*` | tests, gates, proof, terminal claims |
| `docs:*` | documentation, changelog, release notes |
| `review:*` | review posture and review-thread handling |

## Universal Rules

### Authority

- `[authority:entrypoint]` Read the repository's agent entrypoint before work. Common entrypoints are `AGENTS.md`, `CLAUDE.md`, `Claude.md`, `.github/copilot-instructions.md`, and `AUDITOR.md`.
- `[authority:delegation]` If an entrypoint delegates to another file, read the delegated file before acting.
- `[authority:precedence]` User and higher-priority session instructions take precedence. Within the repository, treat machine contracts and enforced gates as authoritative evidence over conflicting explanatory prose; reconcile the conflict rather than ignore it.
- `[authority:prose]` Use prose documentation for orientation when a machine-readable contract exists; do not create a competing rule from prose.
- `[authority:memory]` Do not rely on memory, repo habit, grep alone, or folklore when a routed source exists.
- `[authority:missing]` If a required source, route, contract, fixture or data source is missing, report the exact target and continue independent authorized work. Ask only when the missing input blocks the task.
- `[authority:conflict]` If two instructions conflict, follow the local repository's declared precedence; if precedence is absent, stop and ask the operator.
- `[authority:operator]` Resolve routine implementation choices from task context and evidence. Ask the operator when a material unresolved choice blocks the authorized work.
- `[authority:multi-route]` For a request that spans multiple routed behaviors, decompose the request and load every required route before implementation.

### Scope

- `[scope:one-change]` Keep one PR to one coherent capability, fix, proof obligation, or documentation change.
- `[scope:issue-contract]` If the repository uses slice or issue contracts, the PR must satisfy that contract exactly.
- `[scope:surface-contract]` Keep the diff inside the declared surfaces and outside declared out-of-scope paths.
- `[scope:no-driveby]` Do not include drive-by cleanup, opportunistic refactors, unrelated formatting, or metadata churn.
- `[scope:wrong-contract]` If the issue or slice is wrong, fix the issue or slice before implementation.
- `[scope:runtime-touching]` Runtime-touching work adds proof obligations; it does not replace the primary task contract.
- `[scope:planning]` Planning work publishes scope or backlog state only; it must not mutate runtime behavior.
- `[scope:correctness]` Correctness work fixes existing behavior and must include regression protection.
- `[scope:capability]` Capability work must update the relevant contract, acceptance gate, and end-to-end validation path.
- `[scope:performance]` Performance work must prove semantic equivalence and include metric-delta evidence.

### GitHub

- `[github:branch]` Use the user-selected working branch, otherwise branch from the repository's protected base branch and keep the PR branch up to date before merge.
- `[github:protected-base]` Never push directly to the protected base branch, force-push its history or delete it.
- `[github:worktree]` Use the user-selected branch and an appropriate checkout. Local-only branches and isolated worktrees are valid for authorized local work; do not discard another contributor's changes.
- `[github:commit]` Use Conventional Commits where required: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, or `revert`.
- `[github:commit-scope]` Keep each commit to one logical change.
- `[github:no-ai-attribution]` Do not name an AI, LLM, assistant, model, or model vendor in commit messages or PR titles where the repository bans assistant attribution.
- `[github:pr-title]` If the repository binds PR title to issue title, make them byte-equal.
- `[github:pr-body]` The PR body must identify what changed, why it changed, user or developer impact, and validation.
- `[github:early-pr]` Open the PR as soon as the change is ready for CI when the repository requires early PR flow; continue local verification while CI runs.
- `[github:review-request]` Request the required reviewer when the task authorizes review interaction; a local rule alone is not permission to message another person.
- `[github:review-threads]` All review threads must be answered and resolved before merge.
- `[github:copilot-review]` If the ruleset requires automated code review, the PR must have that review completed.
- `[github:ci]` Read failing check output and fix the named cause. If a gate is wrong, fix the gate in a separate PR.
- `[github:ruleset]` If live branch protection must match a checked-in snapshot, reconcile out-of-band ruleset drift in its own PR.
- `[github:merge]` Merge only after the branch is mergeable, required reviews are satisfied, all required checks are green, and required threads are resolved.

### Change Discipline

- `[change:root-cause]` Fix root causes. Do not paper over missing state with workarounds, fallback paths, or swallowed errors.
- `[change:fail-loud]` Fail early and visibly when required state is absent, contradictory, or unsafe.
- `[change:minimal]` Choose the smallest honest implementation that satisfies the requirement.
- `[change:complexity]` Added complexity must name the specific concern it solves.
- `[change:no-synthetic-data]` Do not invent scientific training or research data. Use documented real fixtures. Generated parser inputs are acceptable for parser property tests and must not stand in for research evidence.
- `[change:expectation]` Validate against the stated expectation, not merely that a command ran.
- `[change:typing]` Do not weaken typing discipline: no new `Any`, broad casts to `Any`, type ignores, pyright ignores, or noqa escapes where banned.
- `[change:exceptions]` Do not add bare exceptions, empty handlers, `contextlib.suppress`, `errors='ignore'`, or semantic equivalents where banned.
- `[change:dependencies]` Prefer the standard library or existing dependencies. Add a dependency only when the task requires it.
- `[change:imports]` Keep imports ordered as standard library, third-party, then local; do not use wildcard imports.
- `[change:logging]` Use module loggers for library code; do not use `print` where the repository bans it.
- `[change:resources]` Use context managers for closeable resources, avoid mutable default arguments, and prefer `pathlib` for paths.
- `[change:api]` Expose public API intentionally and keep internal names clearly internal where the house style requires it.
- `[change:comments]` Add comments only for non-obvious decisions. Do not narrate the line.
- `[change:llm-output]` Model output may draft work, but the contributor owns the final simplified change.

### Evidence

- `[evidence:local-gates]` Run the repository's required local gates before claiming the work is ready.
- `[evidence:exact-candidate]` Terminal claims must be based on the exact candidate pushed to the PR branch.
- `[evidence:ci-authority]` CI is authoritative for merge truth when the repository says so.
- `[evidence:coverage]` Touched authoritative surfaces need routed contract coverage; no coverage means no terminal claim.
- `[evidence:regression]` Every fix or capability must have a regression guard that fails on the old behavior or silent removal.
- `[evidence:e2e]` User-facing capability changes require end-to-end proof when the repository's policy requires it.
- `[evidence:performance]` Performance claims require measured evidence and semantic-equivalence proof.
- `[evidence:terminal-claim]` State local verification separately from remote CI. Claim mergeable only when required CI and live protection pass; a local task may be complete with explicit local evidence.
- `[evidence:report-back]` Conclude with concrete verified changes and reported commit/evidence. Push, open a PR, message reviewers, merge and publish only when authorized by the task.

### Documentation And Release

- `[docs:source]` Repository markdown is the canonical source for documentation unless the repo declares another source of truth.
- `[docs:single-source]` Author a rule, explanation, or procedure once; link to the canonical page instead of copying it.
- `[docs:current-truth]` Document current runnable behavior, not aspirational behavior.
- `[docs:examples]` Examples must be real, runnable, and consistent with implementation.
- `[docs:changelog]` If the repository requires release notes, update `CHANGELOG.md` with an imperative, specific, non-placeholder entry.
- `[docs:version]` If the repository requires versioning, bump SemVer strictly forward according to the change type.
- `[docs:release]` Release notes must be accurate, published in the required format, and not left as drafts.

### Review

- `[review:stance]` Review like a senior engineer: look for semantic risk, not style already enforced by gates.
- `[review:diff-plus]` Treat the diff as the starting point; check callers, consumers, wire formats, configs, docs, and tests affected by the change.
- `[review:findings]` Report only actionable findings naming the path, line, problem and consequence. Post them externally only when the task authorizes review interaction.
- `[review:no-padding]` Do not invent findings or add praise. The opinion is the deliverable.
- `[review:verdict]` Use the repository's verdict ladder or review outcome mapping when one exists.
- `[review:response]` Address every review comment with a commit or an explanation, then resolve the thread when fixed.
- `[review:re-review]` On re-review, check whether prior findings are resolved; do not reopen the whole review unless new changes create new risk.
