# PR review guideline

This is the brief every reviewer of a pull request in this repository follows — GitHub Copilot, an agent, or a human. The repository's hard rules (formatting, imports, type discipline, docstring conventions, dead code, line budgets, coverage, AI-attribution) are enforced mechanically by the CI gates and are documented in `CLAUDE.md`, the constitution. Those are not your job to re-check. This guideline is for the semantic review the gates cannot do: whether the change actually does what it claims, and what it breaks that is not in the diff.

Review like a senior engineer, not a linter. Be meticulous, terse, concrete, and honest. Praise carries no information; cut it. The opinion is the deliverable. When review posting is authorized, place findings inline on the exact file and line. Otherwise prepare the review locally; this guideline does not authorize external messages.

## 0. Before anything else: the attribution stop (BLOCKING)

Scan every commit message between the base and `HEAD`, and the PR title and body, for any mention of an AI or LLM assistant: `claude`, `codex`, `chatgpt`, `gpt-<n>`, `copilot`, `cursor`, `gemini`, `anthropic`, `openai`, `llm`, `ai-assistant`, the phrase "generated with", and any `Co-Authored-By:` trailer naming an assistant. A topical reference to a file literally named `copilot-instructions.md`, or to the repository's required "Copilot review", is not attribution.

Any hit is a blocking failure. Stop and report only that. The author must rewrite history to strip the mention(s) before the PR can move forward. PRs in this repository never carry assistant attribution — this is non-negotiable, and the `pr_checks_cc` gate enforces it server-side, so an unstripped mention will block the merge regardless.

## 1. Read before you judge

Read the PR description and any linked issue first. Know what problem the change solves before you read the diff. Then sit with the diff — understand the change before flagging anything. The difference between genuine help and noise dressed as help is whether you understood the code before judging it.

## 2. Look beyond the diff

This is the core move. The diff shows what the author touched; it does not show what the author forgot. The bug is rarely in the lines that changed — it is in the line three files away that still expects the old shape.

- For every signature change, find every caller.
- For every renamed field, grep every consumer, config file, and doc.
- For every new enum or variant, find every `switch` / `match` / `if`-chain on that type and confirm it handles the new case.
- For every changed wire format or serialized shape, find the other side.

Treat the diff as the starting point for a search, not as the territory.

## 3. What to hunt

- **Correctness** — does the logic do what it claims? Edge cases, off-by-one, null/empty handling.
- **Environmental assumptions** — for each changed function, name what it assumes about disk layout, API shape, network state, and config values; for each assumption, ask what breaks when it is wrong.
- **Identity and namespaces** — if a changed value feeds a dedup key, idempotency token, canonical ID, or partition offset, follow it to every consumer and confirm the new shape cannot collide with other values in the same namespace.
- **Stateful loops** — if a loop has an idempotency guard, mentally execute the second run; every branch that persists data must also update the guard.
- **Destructive operations** — for any `delete` / `reset` / `overwrite` / `truncate`, enumerate every path that reaches it, state the invariant, and verify every exit honors it. Distrust string variables used as control-flow guards.
- **Security surfaces** — for auth, permissions, feature flags, and error handling, trace control flow from the entry point and identify every bypass or short-circuit. Ask: what ships if every TODO is never resolved? If a flag-off path is a security bypass, flag the bypass, not the guarded code.
- **Silent failures** — `try`/`except` that swallows, defaults that mask bugs. The fail-loud gate catches the AST-detectable forms; you catch the semantic ones.
- **Component seams** — across any boundary the PR introduces or changes, trace state flow: who owns mutation, what ordering is assumed, what breaks if one side changes independently.
- **Tests** — present and meaningful. A test that does not exercise the thing it names is worse than no test, because it reads as covered.
- **Cruft** — premature abstraction, speculative flexibility, unused generality, dead code, stale comments, leftover debug output. And bad names — especially bad names.
- **Empirical claims** — a benchmark, throughput, or latency claim with no automated gate to catch its regression is a gap; name it.
- **Docs examples** — verify them against the implementation, never against other docs.

## 4. Discipline before you post

- **Threshold test.** Would a user or a future engineer hit this without warning? If not, do not post it. A comment that is true but not blocking costs the author time and dilutes the ones that matter.
- **Calibrate certainty.** If you are asserting a behavioral fact about the language, runtime, or library and you only pattern-matched it from surface syntax, write "worth verifying" instead of stating it as fact. Confident wrongness costs the author more than silence.
- **Verify the citation.** Read the exact file and line before you flag it. For a pattern spanning files, check every affected file. Never flag already-fixed code.
- **Do not dismiss too fast either.** Before rejecting a finding — yours or another reviewer's — as a misread, read the cited line and trace the failure it claims. If you cannot trace it but it "feels wrong", you are more likely missing context than the finding is wrong.
- **Do not invent findings to look thorough.** If there is nothing real, say so. Padding findings is dishonest and buries the signal.

## 5. The verdict

Reach one overall verdict. Use the four-tier ladder:

- **catastrophe** — shipping this is dangerous: data loss, a security hole, an auth bypass, a crash on a common path, broken for all users. Merge must not happen until it is fixed.
- **bad** — real problems a reviewer must block on. The default state of most code under real scrutiny. Every blocking finding is fixed before merge.
- **mediocre** — it works. Unloved, but survivable. Surface the findings; the author decides.
- **ok** — the rarest outcome. You would sign it off. If there is a single real complaint, it is not ok. There is no tier above ok and no praise above it.

On GitHub, map the verdict to the review event:

- **catastrophe / bad → Request changes**, with at least one inline comment on a specific file and line. Never request changes on the summary alone — if you cannot point to a line, downgrade to a Comment.
- **mediocre → Comment.**
- **ok → Approve.**

`Approve` is the verdict you reach, not the act that merges: merge also requires an eligible non-author approval under the active ruleset. Record the verdict through the review mechanism authorized by the task. The maintainer authority is `mikkokotila`; check live policy before claiming approval unlocks merge.

## 6. How to write a finding

- One finding is `path:line — what is wrong, and why it matters`, in one or two sentences. Concrete enough that a separate agent could act on it without asking what you meant.
- Order findings by severity, not by file order.
- The diagnosis is the deliverable. Propose the exact fix only when you are sure of it, and never let a proposed fix stand in for a clear diagnosis — a complaint you cannot state crisply without attaching a fix is not ready.
- Terse, present tense, no hedging, no softening.

Concrete, actionable findings:

- `src/api.py:34` — the `except` swallows the error and returns `None`; callers cannot tell "no result" from "crashed".
- `worker.py:118` — the retry loop has no backoff; on an upstream outage it will hammer the service.
- `auth.py:52` — `user_id` from the request is compared with `==` against a string; integer IDs silently mismatch.

Vague, unusable findings — do not write these: "error handling is weak" (where?), "this function is too long" (which one?), "the names are bad" (which names?).

## 7. Posting the review on GitHub

- Submit exactly one review event, with every finding as an inline comment on a specific file and line. Do not deliver review feedback as loose issue or PR comments.
- If another reviewer already opened a thread on the same issue, reply on their thread instead of opening a duplicate.
- Before you settle the verdict, clear the ledger: take an explicit position — agree, disagree with a reason, or defer with a reason — on every open thread from another reviewer and every CI check on the PR. A required check that is red and that you cannot explain away means do not approve. "Mostly green" is not a disposition.

## 8. Re-review

When the author says they addressed your feedback, check only whether your previous comments were resolved — do not hunt for new issues on a re-review. If every prior comment is resolved, approve. Otherwise, reply on the unresolved threads stating what is still wrong, and request changes.
