#!/usr/bin/env python3
"""Write the closeout evidence fields into a slice issue's Done Means section.

The `slice_closeout_guard` workflow calls this when a merged PR closes a slice
issue: the merge SHA, the merged PR number and the required CI run ids become
part of the issue body, so a slice's done-ness rests on recorded evidence
rather than on memory.

The splice lives here rather than inline in the workflow because inline
workflow Python cannot be unit tested, and the placement is where this gets
subtly wrong -- the Done Means section match includes the Author Checks header
that terminates it, so appending naively files the evidence under the wrong
heading. `insert_evidence` is pure and the contract tests exercise it directly.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Final

SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r'^##+ Done Means\b.*?^##+ Author Checks\b',
    re.MULTILINE | re.DOTALL,
)
AUTHOR_CHECKS_RE: Final[re.Pattern[str]] = re.compile(r'^##+ Author Checks\b')
BANNER: Final[str] = 'slice_closeout_guard'


def insert_evidence(body: str, merge_sha: str, pr_number: str, runs: list[str]) -> str:
    """Return the issue body with the three evidence fields filled in.

    Fields already present are overwritten; absent ones are inserted before the
    Author Checks header. Raises `ValueError` when the Done Means section
    cannot be located unambiguously -- that is contradicted structure rather
    than absent scaffolding, and the guard fails closed on it.
    """
    matches = list(SECTION_RE.finditer(body))
    if len(matches) != 1:
        raise ValueError(
            f'expected exactly one Done Means section, found {len(matches)}'
        )
    start, end = matches[0].span()
    lines = matches[0].group(0).split('\n')

    out: list[str] = []
    found_sha = found_pr = found_runs = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('Merge SHA:'):
            out.append(f'Merge SHA: {merge_sha}')
            found_sha = True
        elif line.startswith('Merged PR number:'):
            out.append(f'Merged PR number: #{pr_number}')
            found_pr = True
        elif line.startswith('Required CI runs'):
            out.append(line)
            found_runs = True
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith('-'):
                i += 1
            out.extend(runs)
            continue
        else:
            out.append(line)
        i += 1

    # Absent fields are missing scaffolding, not contradicted evidence: the
    # merge SHA, the PR number and the run ids are all in hand, and only the
    # lines to hold them are missing. Refusing here reopened correctly merged
    # slices over body formatting.
    appended: list[str] = []
    if not found_sha:
        appended.append(f'Merge SHA: {merge_sha}')
    if not found_pr:
        appended.append(f'Merged PR number: #{pr_number}')
    if not found_runs:
        appended.append('Required CI runs (workflow name : run id):')
        appended.extend(runs)
    if appended:
        # The matched section ends with the Author Checks header, so the fields
        # go before it -- and before the blank line separating them -- not at
        # the end of `out`.
        tail = len(out) - 1
        while tail > 0 and not AUTHOR_CHECKS_RE.match(out[tail]):
            tail -= 1
        while tail > 0 and not out[tail - 1].strip():
            tail -= 1
        out[tail:tail] = [*appended, '']

    return body[:start] + '\n'.join(out) + body[end:]


def main() -> int:
    """Read the guard's staged files, splice the evidence, write the new body."""
    body = Path('issue_body.txt').read_text(encoding='utf-8')
    required = [
        c for c in Path('required_contexts.txt').read_text(encoding='utf-8').splitlines() if c
    ]
    if not required:
        print(
            f'{BANNER}: no required status checks resolved from the main branch rules',
            file=sys.stderr,
        )
        return 1

    check_runs: dict[str, str] = {}
    for raw in Path('check_runs.txt').read_text(encoding='utf-8').splitlines():
        if not raw:
            continue
        name, _, run_id = raw.partition('\t')
        check_runs.setdefault(name, run_id)

    missing = [c for c in required if c not in check_runs]
    if missing:
        print(
            f'{BANNER}: required contexts without a successful check run on the '
            f"closing PR head commit: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    runs = [f'- {c} : {check_runs[c]}' for c in required]
    try:
        new_body = insert_evidence(
            body, os.environ['MERGE_SHA'], os.environ['PR_NUMBER'], runs
        )
    except ValueError as exc:
        print(f'{BANNER}: {exc}', file=sys.stderr)
        return 1

    Path('new_body.txt').write_text(new_body, encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
