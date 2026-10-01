#!/usr/bin/env python3
"""Coverage ratchet gate: the coverage floor can only be lowered with a marker.

The mechanical twin of `check_budget_ratchet.py`, inverted. A module
budget is a ceiling, so *raising* it is the loosening that needs a marker;
the coverage floor in `.github/budgets.json` is a floor, so
*lowering* it is the loosening. Either field (`line`, `branch`) that drops
below its value at the PR base must be authorised by a PR-body line:

    [coverage-lower: line: <reason>]
    [coverage-lower: branch: <reason>]

so the oracle cannot be quietly weakened by the same PR that would then
slip under it. Raising the floor needs no marker -- that is the ratchet
turning the right way.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Final

from _common import REPO_ROOT, fail_setup

HEAD_BUDGET_PATH = REPO_ROOT / '.github' / 'budgets.json'
BUDGET_SECTION = 'coverage'
FIELDS: Final[tuple[str, ...]] = ('line', 'branch')

LOWER_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r'^\[coverage-lower:\s*(?P<field>line|branch):\s*(?P<reason>.*?\S)\s*\]\s*$',
    re.MULTILINE,
)


# Bind this gate's banner to the shared setup-failure reporter.
_fail_setup = partial(fail_setup, 'COVERAGE RATCHET GATE')


def _parse_floor(text: str) -> dict[str, int]:
    if not text.strip():
        return {}
    try:
        data = json.loads(text).get(BUDGET_SECTION, {})
    except json.JSONDecodeError as exc:
        _fail_setup(f'cannot parse coverage_budget JSON: {exc}')
    if not isinstance(data, dict):
        _fail_setup('coverage_budget JSON is not an object')
    parsed: dict[str, int] = {}
    for field in FIELDS:
        value = data.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
            _fail_setup(f'coverage_budget["{field}"] must be an int in [0, 100], got {value!r}')
        parsed[field] = int(value)
    return parsed


def _base_floor_from_ref(base_ref: str) -> dict[str, int]:
    cmd = ['git', 'show', f'{base_ref}:.github/budgets.json']
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {}
    return _parse_floor(result.stdout) if result.returncode == 0 else {}


def _pr_body_from_number(pr_number: int) -> str:
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN', '')
    if not token:
        _fail_setup('GH_TOKEN/GITHUB_TOKEN required for --pr-number mode')
    owner_repo = os.environ.get('GITHUB_REPOSITORY', 'autonomio/talos')
    cmd = ['gh', 'api', f'repos/{owner_repo}/pulls/{pr_number}', '--jq', '.body']
    result = subprocess.run(cmd, check=False, capture_output=True, text=True,
                            env={**os.environ, 'GH_TOKEN': token})
    if result.returncode != 0:
        _fail_setup(f'gh api failed: {result.stderr.strip()}')
    return result.stdout


def check(base: dict[str, int], head: dict[str, int], pr_body: str) -> list[tuple[str, int, int]]:
    drops = [(f, base[f], head[f]) for f in FIELDS if f in base and f in head and head[f] < base[f]]
    declared = {m.group('field') for m in LOWER_MARKER_RE.finditer(pr_body)}
    return [(f, b, h) for f, b, h in drops if f not in declared]


def main() -> int:
    ap = argparse.ArgumentParser(description='Coverage ratchet gate')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--base-ref', help='git ref of PR base (CI mode)')
    mode.add_argument('--base-file', help='path to base coverage_budget JSON (local/test mode)')
    ap.add_argument('--pr-number', type=int, help='PR number; requires GH_TOKEN')
    ap.add_argument('--pr-body-file', help='path to file with PR body text')
    args = ap.parse_args()
    if args.base_ref is not None:
        if args.pr_number is None:
            ap.error('--base-ref requires --pr-number (CI mode)')
        base = _base_floor_from_ref(args.base_ref)
        pr_body = _pr_body_from_number(int(args.pr_number))
    else:
        if args.pr_body_file is None:
            ap.error('--base-file requires --pr-body-file (local mode)')
        base_path = Path(args.base_file)
        base = _parse_floor(base_path.read_text(encoding='utf-8')) if base_path.is_file() else {}
        body_path = Path(args.pr_body_file)
        pr_body = body_path.read_text(encoding='utf-8') if body_path.is_file() else ''
    if not base:
        print('COVERAGE RATCHET GATE -- PASS (vacuous: base has no coverage budget)')
        return 0
    head = _parse_floor(HEAD_BUDGET_PATH.read_text(encoding='utf-8')) if HEAD_BUDGET_PATH.is_file() else {}
    violations = check(base, head, pr_body)
    if violations:
        print('COVERAGE RATCHET GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for field, b, h in violations:
            print(f'  lowered without marker: {field} floor (base={b}%, head={h}%, {h - b})',
                  file=sys.stderr)
        print('', file=sys.stderr)
        print('  PR body must contain `[coverage-lower: <field>: <reason>]` on its own line',
              file=sys.stderr)
        print(f'  for EACH lowered field. {len(violations)} violation(s). Merge blocked.',
              file=sys.stderr)
        return 1
    print('COVERAGE RATCHET GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
