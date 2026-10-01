#!/usr/bin/env python3
"""Coverage floor gate: actual coverage must clear a ratcheting floor.

Two rules over `talos/`, read from `coverage.json`:

  FLOOR  actual line/branch coverage >= the floor in
         `.github/budgets.json`. The Limen-style absolute gate.
  TRACK  once the package is non-trivial, the floor may not lag actual
         coverage by more than TRACK_SLACK points -- a real improvement
         must be banked into the floor so it cannot silently erode back.

FLOOR alone lets a 95%-covered package rot to 51% under a static 50%
floor. TRACK closes that slack: gains are locked in. The floor can only
be *lowered* through `check_coverage_ratchet.py` (with a PR-body marker),
so the two rules together make the floor a genuine ratchet.
"""
from __future__ import annotations

import json
import math
import sys
from functools import partial
from typing import Final

from _common import REPO_ROOT, exit_if_disabled, fail_setup, gate_setting

COVERAGE_JSON = REPO_ROOT / 'coverage.json'
BUDGET_PATH = REPO_ROOT / '.github' / 'budgets.json'
BUDGET_SECTION = 'coverage'

# TRACK engages only once there is enough code for the percentage to be a
# stable signal -- a 1-statement stub at a vacuous 100% must not be forced
# to a 98% floor that breaks the first real, partially-tested module.
BANNER: Final[str] = 'COVERAGE FLOOR'
DEFAULT_MIN_STATEMENTS: Final[int] = 50
DEFAULT_MIN_BRANCHES: Final[int] = 20
DEFAULT_TRACK_SLACK: Final[int] = 2


# Bind this gate's banner to the shared setup-failure reporter.
_fail_setup = partial(fail_setup, 'COVERAGE FLOOR GATE')


def _num(totals: dict[str, object], key: str) -> float:
    value = totals.get(key, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _load_floor() -> tuple[int, int]:
    if not BUDGET_PATH.is_file():
        _fail_setup(f'missing {BUDGET_PATH.relative_to(REPO_ROOT)} (the coverage floor)')
    try:
        raw = json.loads(BUDGET_PATH.read_text(encoding='utf-8')).get(BUDGET_SECTION, {})
    except (OSError, json.JSONDecodeError) as exc:
        _fail_setup(f'cannot read {BUDGET_PATH}: {exc}')
    if not isinstance(raw, dict):
        _fail_setup(f'{BUDGET_PATH} is not a JSON object')
    out: list[int] = []
    for key in ('line', 'branch'):
        value = raw.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
            _fail_setup(f'{BUDGET_PATH}["{key}"] must be an int in [0, 100], got {value!r}')
        out.append(int(value))
    return out[0], out[1]


def _line_pct(totals: dict[str, object]) -> float:
    raw = totals.get('percent_statements_covered', totals.get('percent_covered', 0.0))
    return float(raw) if isinstance(raw, (int, float)) else 0.0


def _branch_pct(totals: dict[str, object]) -> float:
    if int(_num(totals, 'num_branches')) == 0:
        return 100.0  # no branches -> branch coverage is vacuously complete
    raw = totals.get('percent_branches_covered', totals.get('percent_covered', 0.0))
    return float(raw) if isinstance(raw, (int, float)) else 0.0


def _track_violation(
    label: str, actual: float, floor: int, units: int, min_units: int, slack: int
) -> str | None:
    if units < min_units:
        return None
    banked = math.floor(actual) - slack
    if banked <= floor:
        return None
    return (
        f'{label} coverage is {actual:.1f}% but the floor is still {floor}%. '
        f'Bank the gain: raise coverage."{label}" in budgets.json to >= {banked}%.'
    )


def main() -> int:
    exit_if_disabled('coverage', BANNER)
    if not COVERAGE_JSON.is_file():
        _fail_setup(f'no coverage.json at {COVERAGE_JSON} (run `coverage json` first)')
    data = json.loads(COVERAGE_JSON.read_text(encoding='utf-8'))
    totals = data.get('totals', {})
    num_statements = int(_num(totals, 'num_statements'))
    if num_statements == 0:
        _fail_setup('coverage.json reports 0 statements under the package root '
                    '(coverage likely ran over an empty or wrong --source)')
    num_branches = int(_num(totals, 'num_branches'))
    line_pct = _line_pct(totals)
    branch_pct = _branch_pct(totals)
    line_floor, branch_floor = _load_floor()

    failures: list[str] = []
    if line_pct < line_floor:
        failures.append(f'line coverage:   {line_pct:.1f}% (floor {line_floor}%)')
    if branch_pct < branch_floor:
        failures.append(f'branch coverage: {branch_pct:.1f}% (floor {branch_floor}%)')
    min_stmts = gate_setting('coverage', 'min_statements_for_track', DEFAULT_MIN_STATEMENTS, BANNER)
    min_brs = gate_setting('coverage', 'min_branches_for_track', DEFAULT_MIN_BRANCHES, BANNER)
    slack = gate_setting('coverage', 'track_slack', DEFAULT_TRACK_SLACK, BANNER)
    for msg in (
        _track_violation('line', line_pct, line_floor, num_statements, min_stmts, slack),
        _track_violation('branch', branch_pct, branch_floor, num_branches, min_brs, slack),
    ):
        if msg is not None:
            failures.append(msg)

    if failures:
        print('COVERAGE FLOOR GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for line in failures:
            print(f'  {line}', file=sys.stderr)
        missing = sorted(
            (rel, f['missing_lines'])
            for rel, f in (data.get('files') or {}).items()
            if f.get('missing_lines')
        )[:10]
        if missing:
            print('', file=sys.stderr)
            print('  missing lines (top 10 files):', file=sys.stderr)
            for rel, lines in missing:
                preview = ','.join(str(n) for n in lines[:5])
                print(f'    {rel}: {preview}{"…" if len(lines) > 5 else ""}', file=sys.stderr)
        print('', file=sys.stderr)
        print('Merge blocked.', file=sys.stderr)
        return 1
    print(f'COVERAGE FLOOR GATE -- PASS (line={line_pct:.1f}%>={line_floor}%, '
          f'branch={branch_pct:.1f}%>={branch_floor}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
