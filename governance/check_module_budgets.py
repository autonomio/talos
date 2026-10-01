#!/usr/bin/env python3
"""Module line-count budget gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import REPO_ROOT, fail_setup, resolve_package_dir, significant_lines

BUDGET_PATH = REPO_ROOT / '.github' / 'budgets.json'
BUDGET_SECTION = 'modules'
BANNER = 'MODULE BUDGET GATE'


def load_budgets() -> dict[str, int]:
    if not BUDGET_PATH.is_file():
        fail_setup(BANNER, f'missing budget file: {BUDGET_PATH}')
    try:
        raw = json.loads(BUDGET_PATH.read_text(encoding='utf-8')).get(BUDGET_SECTION, {})
    except (json.JSONDecodeError, OSError) as exc:
        fail_setup(BANNER, f'cannot read {BUDGET_PATH}: {exc}')
    if not isinstance(raw, dict):
        fail_setup(BANNER, f'{BUDGET_PATH} is not a JSON object')
    parsed: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(value, int) or value <= 0:
            fail_setup(BANNER, f'{BUDGET_PATH}["{key}"] must be positive int, got {value!r}')
        parsed[str(key)] = value
    return parsed


def check(budgets: dict[str, int], source_dir: Path) -> tuple[list[tuple[str, int, int]], list[str]]:
    # Returns (over_budget, unbudgeted_source_paths). Any path under the
    # package root that is NOT declared under `modules` in budgets.json is a
    # violation -- otherwise a new module could silently escape the gate.
    over: list[tuple[str, int, int]] = []
    for rel_path, budget in budgets.items():
        path = REPO_ROOT / rel_path
        if not path.is_file():
            continue
        actual = significant_lines(path)
        if actual > budget:
            over.append((rel_path, actual, budget))
    package_prefix = f'{source_dir.name}/'
    declared = {key for key in budgets if key.startswith(package_prefix)}
    unbudgeted: list[str] = []
    for path in sorted(source_dir.rglob('*.py')):
        rel = str(path.relative_to(REPO_ROOT))
        if rel not in declared:
            unbudgeted.append(rel)
    return over, unbudgeted


def main() -> int:
    budgets = load_budgets()
    source_dir = resolve_package_dir(BANNER)
    over, unbudgeted = check(budgets, source_dir)
    if over or unbudgeted:
        print('MODULE BUDGET GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for rel_path, actual, budget in over:
            overage = actual - budget
            print(
                f'  - {rel_path}: {actual} lines '
                f'(budget {budget}, overage +{overage})',
                file=sys.stderr,
            )
        for rel_path in unbudgeted:
            print(
                f'  - {rel_path}: undeclared module; '
                f'add an entry under "modules" in .github/budgets.json',
                file=sys.stderr,
            )
        print('', file=sys.stderr)
        total = len(over) + len(unbudgeted)
        print(f'{total} violation(s). Merge blocked.', file=sys.stderr)
        return 1
    print('MODULE BUDGET GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
