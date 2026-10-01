#!/usr/bin/env python3
"""Budget ratchet gate: budgets can only be lowered without PR-body marker."""
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
BUDGET_SECTION = 'modules'

RAISE_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r'^\[budget-raise:\s*(?P<path>[^:\]]+):\s*(?P<reason>.*?\S)\s*\]\s*$',
    re.MULTILINE,
)

# Bind this gate's banner to the shared setup-failure reporter.
_fail_setup = partial(fail_setup, 'BUDGET RATCHET GATE')


def _parse_budget(text: str) -> dict[str, int]:
    if not text.strip():
        return {}
    try:
        data = json.loads(text).get(BUDGET_SECTION, {})
    except json.JSONDecodeError as exc:
        _fail_setup(f'cannot parse budget JSON: {exc}')
    if not isinstance(data, dict):
        _fail_setup('budget JSON is not an object')
    parsed: dict[str, int] = {}
    for key, value in data.items():
        if not isinstance(value, int) or value <= 0:
            _fail_setup(f'budget["{key}"] must be positive int, got {value!r}')
        parsed[str(key)] = value
    return parsed


def _base_budget_from_ref(base_ref: str) -> dict[str, int]:
    cmd = ['git', 'show', f'{base_ref}:.github/budgets.json']
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {}
    if result.returncode != 0:
        return {}
    return _parse_budget(result.stdout)


def _pr_body_from_number(pr_number: int) -> str:
    token = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN', '')
    if not token:
        print('BUDGET RATCHET GATE -- FAIL', file=sys.stderr)
        print('  GH_TOKEN/GITHUB_TOKEN required for --pr-number mode', file=sys.stderr)
        sys.exit(2)
    owner_repo = os.environ.get('GITHUB_REPOSITORY', 'autonomio/talos')
    cmd = ['gh', 'api', f'repos/{owner_repo}/pulls/{pr_number}', '--jq', '.body']
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env={**os.environ, 'GH_TOKEN': token})
    if result.returncode != 0:
        print('BUDGET RATCHET GATE -- FAIL', file=sys.stderr)
        print(f'  gh api failed: {result.stderr.strip()}', file=sys.stderr)
        sys.exit(2)
    return result.stdout


def check(base: dict[str, int], head: dict[str, int], pr_body: str) -> list[tuple[str, int, int]]:
    raises = [(p, base[p], head[p]) for p in head if p in base and head[p] > base[p]]
    if not raises:
        return []
    declared_paths = {m.group('path').strip() for m in RAISE_MARKER_RE.finditer(pr_body)}
    return [(p, b, h) for p, b, h in raises if p not in declared_paths]


def main() -> int:
    ap = argparse.ArgumentParser(description='Budget ratchet gate')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--base-ref', help='git ref of PR base (CI mode)')
    mode.add_argument('--base-file', help='path to base budget JSON (local/test mode)')
    ap.add_argument('--pr-number', type=int, help='PR number; requires GH_TOKEN')
    ap.add_argument('--pr-body-file', help='path to file with PR body text')
    args = ap.parse_args()
    if args.base_ref is not None:
        if args.pr_number is None:
            ap.error('--base-ref requires --pr-number (CI mode)')
        if args.pr_body_file is not None:
            ap.error('cannot mix --base-ref (CI mode) with --pr-body-file (local mode)')
    else:
        if args.pr_body_file is None:
            ap.error('--base-file requires --pr-body-file (local mode)')
        if args.pr_number is not None:
            ap.error('cannot mix --base-file (local mode) with --pr-number (CI mode)')
    if args.base_ref is not None:
        base = _base_budget_from_ref(args.base_ref)
        pr_body = _pr_body_from_number(int(args.pr_number))
    else:
        base_path = Path(args.base_file)
        base = _parse_budget(base_path.read_text(encoding='utf-8')) if base_path.is_file() and base_path.stat().st_size > 0 else {}
        body_path = Path(args.pr_body_file)
        pr_body = body_path.read_text(encoding='utf-8') if body_path.is_file() and body_path.stat().st_size > 0 else ''
    if not base:
        print('BUDGET RATCHET GATE -- PASS (vacuous: base has no budget file)')
        return 0
    head = _parse_budget(HEAD_BUDGET_PATH.read_text(encoding='utf-8')) if HEAD_BUDGET_PATH.is_file() else {}
    violations = check(base, head, pr_body)
    if violations:
        print('BUDGET RATCHET GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for path, b, h in violations:
            print(f'  raised without marker: {path} (base={b}, head={h}, +{h - b})', file=sys.stderr)
        print('', file=sys.stderr)
        print('  PR body must contain `[budget-raise: <path>: <reason>]` on its own line', file=sys.stderr)
        print(f'  for EACH raised path. {len(violations)} violation(s). Merge blocked.', file=sys.stderr)
        return 1
    print('BUDGET RATCHET GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
