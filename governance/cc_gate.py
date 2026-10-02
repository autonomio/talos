#!/usr/bin/env python3
"""Conventional Commits gate -- hard-fail on any deviation.

The linked-issue rule checks every closing-referenced issue rather
than short-circuiting on a single reference: under slice-gate rule 9
the last slice's PR closes {slice, parent PRD}, and a single-reference
short-circuit would skip slice-title validation exactly there. Only
parent PRDs — issues carrying the ``planning`` label — are exempt,
because the PRD title format is not a Conventional Commits subject.

This gate enforces the Conventional Commits v1.0.0 specification on:

  1. The PR title.
  2. The title of every issue the PR closes except ``planning``-labelled
     parent PRDs (resolved by the same Closes/Fixes/Resolves #N rule
     the slice gate uses; reference-count discipline and unresolvable
     references are the slice gate's concern, not duplicated here).
  3. Every non-merge commit message in the PR's commit range
     (``$BASE..$HEAD``).

It additionally fails if the PR title or any non-merge commit message
names an AI/LLM assistant -- commit metadata in this org never carries
AI attribution.

Specification reference: https://www.conventionalcommits.org/en/v1.0.0/

Accepted format:

    <type>[optional scope][!]: <description>

where ``<type>`` is one of::

    feat | fix | docs | style | refactor | perf | test | build |
    ci | chore | revert

all lowercase. Optional ``scope`` is parenthesized lowercase
alphanumeric (hyphens, underscores, slashes, dots allowed). Optional ``!`` before
the colon marks a breaking change. ``<description>`` must be
non-empty after the ``:`` plus exactly one space.

Exit codes:
  0 -- every checked subject matches CC.
  1 -- at least one deviation.
  2 -- gate setup failure (bad args, git/gh failure, etc.).

Usage:

  python governance/cc_gate.py \\
    --pr-title "<pr title>" \\
    --pr-body-file <path> \\
    --base-ref <rev> \\
    --head-ref <rev> \\
    --repo <owner>/<name>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

from _common import CC_RE, CLOSING_KEYWORD_RE, comparison_ref, exit_if_disabled, section_setting

# git log --format=%H%x09%P%x09%s yields exactly three tab-separated fields.
GIT_LOG_FIELD_COUNT: Final[int] = 3

# Allowed types per the widely-followed set (Angular convention + CC v1.0.0).
# The default; `commits.types` in governance.yml overrides it, because a
# derived repository narrowing the set is a legitimate thing to want and the
# section declaring it was previously read by nothing.
BANNER: Final[str] = 'CC GATE'

DEFAULT_CC_TYPES: Final[frozenset[str]] = frozenset({
    'feat', 'fix', 'docs', 'style', 'refactor', 'perf',
    'test', 'build', 'ci', 'chore', 'revert',
})


def cc_types() -> frozenset[str]:
    """The Conventional Commits types this repository accepts."""
    return frozenset(
        section_setting('commits', 'types', sorted(DEFAULT_CC_TYPES), BANNER)
    )


# AI/LLM attribution scan. Commit metadata in this org never names an
# AI/LLM assistant, so the PR title and every non-merge commit message
# must be free of these markers. Legitimate topical references are scrubbed
# first (ATTRIBUTION_EXEMPT_RE): the repo's own *.md governance files
# (CLAUDE.md, AGENTS.md, copilot-instructions.md) and the required GitHub
# Copilot review -- those are filenames and a feature, not authorship.
ATTRIBUTION_EXEMPT_RE: Final[re.Pattern[str]] = re.compile(
    r'\b[\w-]+\.md\b|\bcopilot(?:[ -]code)?[ -]review\b',
    re.IGNORECASE,
)
# The bare tokens ``gemini``, ``cursor``, ``llm``, and unqualified
# ``generated with`` collide with legitimate domain vocabulary (the
# Gemini crypto exchange, database cursors, "generated with <tool>")
# and would hard-fail a required gate on text that names no AI
# assistant. They are narrowed to AI-qualified forms; every unambiguous
# marker stays bare. ``api`` and ``code`` are deliberately not Gemini
# qualifiers — "Gemini API client" is exchange vocabulary — and the
# co-author alternation is word-bounded so a surname like Hillman
# cannot match ``llm`` as a substring.
ATTRIBUTION_RE: Final[re.Pattern[str]] = re.compile(
    r'\bclaude\b|\bcodex\b|\bchatgpt\b|\bgpt-?\d\b|\bcopilot\b'
    r'|\bgoogle[ -]gemini\b|\bgemini[ -](?:pro|ultra|flash|cli)\b'
    r'|\bcursor[ -](?:ai|ide|agent|editor)\b'
    r'|\banthropic\b|\bopenai\b|\bai[ -]?assistant\b'
    r'|\bllm[ -](?:assist(?:ant|ed)|generated|written|authored)\b'
    r'|generated[ -]with[ -](?:an?[ -])?(?:claude|chatgpt|codex|copilot|cursor|gemini|gpt|llm|ai)\b'
    r'|co-authored-by:\s*.*\b(?:claude|openai|anthropic|chatgpt|codex|copilot|cursor|gemini|gpt|llm)\b',
    re.IGNORECASE,
)


def check_cc(subject: str) -> str | None:
    """Return ``None`` if ``subject`` matches CC; otherwise return a
    human-readable reason for the failure."""
    first_line = subject.split('\n', 1)[0]
    if not first_line:
        return 'empty subject'
    match = CC_RE.match(first_line)
    if match is None:
        return (
            'does not match Conventional Commits v1.0.0 format '
            "'<type>[(scope)][!]: <description>'"
        )
    cc_type = match.group('type')
    if cc_type not in cc_types():
        return (
            f"type {cc_type!r} is not one of the allowed CC types "
            f"(allowed: {sorted(cc_types())!r})"
        )
    description = match.group('description')
    if not description or not description.strip():
        return 'description after the colon is empty or whitespace-only'
    return None


def run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ['git', *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'cc_gate: git not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
    if result.returncode != 0:
        print(
            f'cc_gate: git {" ".join(args)} failed (exit {result.returncode}): '
            f'{result.stderr.strip()}',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return result.stdout


def list_commits(base_ref: str, head_ref: str) -> list[dict[str, object]]:
    """List every commit in ``base_ref..head_ref`` with metadata for
    deciding whether to check it. ``is_merge`` is derived from the
    parent count (more than one = merge commit)."""
    out = run_git([
        'log', f'{base_ref}..{head_ref}',
        '--format=%H%x09%P%x09%s',
    ])
    commits: list[dict[str, object]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split('\t', 2)
        if len(parts) != GIT_LOG_FIELD_COUNT:
            # Fail closed: a line this gate cannot parse means commits
            # could go unchecked, which is a setup failure, not a skip.
            print(
                f'cc_gate: unparseable git log line: {line!r}',
                file=sys.stderr,
            )
            raise SystemExit(2)
        sha, parents, subject = parts
        is_merge = len(parents.split()) > 1
        commits.append({
            'sha': sha,
            'subject': subject,
            'is_merge': is_merge,
        })
    return commits


def fetch_issue(repo: str, number: int) -> dict[str, object]:
    """Return ``{'title': ..., 'labels': [...]}`` for the issue.

    Any gh failure here is a gate setup failure because linked-issue
    title validation is part of cc_gate's own contract.
    """
    try:
        result = subprocess.run(
            [
                'gh', 'api', f'repos/{repo}/issues/{number}',
                '--jq', '{title: .title, labels: [.labels[].name]}',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'cc_gate: gh not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc
    if result.returncode != 0:
        error_output = (
            result.stderr.strip()
            or result.stdout.strip()
            or 'no error output'
        )
        print(
            f'cc_gate: gh api repos/{repo}/issues/{number} failed '
            f'(exit {result.returncode}): {error_output}',
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(
            f'cc_gate: linked issue #{number} payload is not JSON: {exc}',
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    if not isinstance(payload, dict) or not str(payload.get('title', '')).strip():
        print(
            f'cc_gate: linked issue #{number} has an empty title payload',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return payload


def find_closing_references(body: str) -> list[int]:
    # Deduplicated, first-seen order: a body repeating a reference must
    # not fetch the same issue twice.
    return list(dict.fromkeys(
        int(m.group(1)) for m in CLOSING_KEYWORD_RE.finditer(body)
    ))


def attribution_hit(text: str) -> str | None:
    """Return the first AI/LLM-attribution substring in ``text``, or None.

    Legitimate topical references (``*.md`` filenames, the Copilot review
    feature) are scrubbed before the scan so they cannot register.
    """
    scrubbed = ATTRIBUTION_EXEMPT_RE.sub(' ', text)
    match = ATTRIBUTION_RE.search(scrubbed)
    if match is None:
        return None
    return match.group(0)


def list_commit_messages(base_ref: str, head_ref: str) -> list[tuple[str, str]]:
    """Return ``(sha, full message)`` for every non-merge commit in range.

    Merge commits are excluded (``--no-merges``) so an auto-generated merge
    subject that happens to name a branch like ``copilot/...`` does not
    register as authored attribution.
    """
    out = run_git([
        'log', f'{base_ref}..{head_ref}', '--no-merges', '--format=%H%x1f%B%x1e',
    ])
    messages: list[tuple[str, str]] = []
    for record in out.split('\x1e'):
        stripped = record.strip()
        if not stripped:
            continue
        sha, _, body = stripped.partition('\x1f')
        messages.append((sha.strip(), body))
    return messages


def gate(
    pr_title: str,
    pr_body: str,
    base_ref: str,
    head_ref: str,
    repo: str,
) -> list[str]:
    failures: list[str] = []

    # Rule 1: PR title.
    err = check_cc(pr_title)
    if err is not None:
        failures.append(f'PR title {pr_title!r} {err}.')

    # Rule 2: every non-merge commit in range.
    commits = list_commits(base_ref, head_ref)
    for c in commits:
        if c['is_merge']:
            continue
        sha = str(c['sha'])
        subject = str(c['subject'])
        err = check_cc(subject)
        if err is not None:
            failures.append(f'commit {sha[:8]} subject {subject!r} {err}.')

    # Rule 3: linked issue titles. Reference-count discipline is a
    # slice_gate concern; cc_gate checks CC on every closing-referenced
    # issue and exempts only ``planning``-labelled parent PRDs — an
    # exemption scoped to any non-slice label would silently drop the
    # validation the old single-reference branch applied to every
    # ordinary issue.
    refs = find_closing_references(pr_body)
    for number in refs:
        issue = fetch_issue(repo, number)
        labels = issue.get('labels')
        if isinstance(labels, list) and 'planning' in labels:
            continue
        issue_title = str(issue.get('title', ''))
        err = check_cc(issue_title)
        if err is not None:
            failures.append(
                f'linked issue #{number} title {issue_title!r} {err}.'
            )

    # Rule 4: no AI/LLM attribution in the PR title or any commit message.
    title_hit = attribution_hit(pr_title)
    if title_hit is not None:
        failures.append(
            f'PR title names an AI/LLM assistant ({title_hit!r}); strip the attribution.'
        )
    for sha, message in list_commit_messages(base_ref, head_ref):
        hit = attribution_hit(message)
        if hit is not None:
            failures.append(
                f'commit {sha[:8]} message names an AI/LLM assistant '
                f'({hit!r}); strip the attribution.'
            )

    return failures


def main() -> int:
    exit_if_disabled('cc', BANNER)
    parser = argparse.ArgumentParser(description='Conventional Commits gate')
    parser.add_argument('--pr-title', required=True)
    parser.add_argument('--pr-body-file', required=True)
    parser.add_argument('--base-ref', required=True)
    parser.add_argument('--head-ref', required=True)
    parser.add_argument('--repo', required=True)
    args = parser.parse_args()

    try:
        pr_body = Path(args.pr_body_file).read_text(encoding='utf-8')
    except OSError as exc:
        print(
            f'cc_gate: cannot read --pr-body-file {args.pr_body_file}: {exc}',
            file=sys.stderr,
        )
        return 2

    failures = gate(
        args.pr_title,
        pr_body,
        comparison_ref(args.base_ref, args.head_ref),
        args.head_ref,
        args.repo,
    )

    if failures:
        print('CC GATE -- FAIL')
        print()
        for msg in failures:
            print(f'  - {msg}')
        print()
        print(f'{len(failures)} failure(s). Merge blocked.')
        return 1

    print('CC GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
