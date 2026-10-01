#!/usr/bin/env python3
"""Diff coverage gate: changed executable lines in the package must be covered."""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Final

from _common import (
    REPO_ROOT,
    comparison_ref,
    fail_setup,
    gate_setting,
    resolve_package_dir,
)

COVERAGE_JSON = REPO_ROOT / 'coverage.json'

# Floor for CHANGED package lines specifically -- higher than the global
# coverage floor, because new code must arrive tested. Raise toward 100 as
# the package matures.
DEFAULT_DIFF_FLOOR: Final[float] = 80.0
BANNER = 'DIFF COVERAGE GATE'


def parse_added_lines(diff_text: str) -> dict[str, set[int]]:
    """Map each file to the set of new-file line numbers it ADDS.

    Parses `git diff -U0` output: every hunk header `@@ -a,b +c,d @@` resets
    the new-file line counter to c, then each `+` line records and advances it
    while `-` lines (removed) are ignored.
    """
    added: dict[str, set[int]] = {}
    current: str | None = None
    new_line = 0
    for line in diff_text.splitlines():
        if line.startswith('+++ b/'):
            current = line[len('+++ b/'):]
            added.setdefault(current, set())
        elif line.startswith('@@'):
            plus = line.split('+', 1)[1].split(' ', 1)[0]
            new_line = int(plus.split(',', 1)[0])
        elif current is not None and line.startswith('+') and not line.startswith('+++'):
            added[current].add(new_line)
            new_line += 1
    return {path: lines for path, lines in added.items() if lines}


def evaluate(
    added: dict[str, set[int]],
    files: dict[str, object],
) -> tuple[int, int, list[tuple[str, list[int]]]]:
    """Return (changed_executable, changed_covered, uncovered-by-file)."""
    total = 0
    covered = 0
    uncovered: list[tuple[str, list[int]]] = []
    for rel, lines in added.items():
        info = files.get(rel)
        if not isinstance(info, dict):
            continue
        executed = {int(n) for n in info.get('executed_lines', [])}
        missing = {int(n) for n in info.get('missing_lines', [])}
        executable = lines & (executed | missing)
        if not executable:
            continue
        miss = sorted(lines & missing)
        total += len(executable)
        covered += len(executable) - len(miss)
        if miss:
            uncovered.append((rel, miss))
    return total, covered, uncovered


def _package_root() -> str:
    return resolve_package_dir('DIFF COVERAGE GATE').relative_to(REPO_ROOT).as_posix()


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != '--base-ref':
        fail_setup(BANNER, 'usage: check_diff_coverage.py --base-ref <ref>')
    base_ref = comparison_ref(sys.argv[2])
    package_root = _package_root()
    if not COVERAGE_JSON.is_file():
        fail_setup(BANNER, f'no coverage.json at {COVERAGE_JSON}')
    result = subprocess.run(
        ['git', 'diff', '-U0', f'{base_ref}...HEAD', '--', package_root],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        fail_setup(BANNER, f'git diff failed: {result.stderr.strip()}')
    added = parse_added_lines(result.stdout)
    files = json.loads(COVERAGE_JSON.read_text(encoding='utf-8')).get('files', {})
    total, covered, uncovered = evaluate(added, files)
    if total == 0:
        print('DIFF COVERAGE GATE -- PASS (no changed executable lines in the package)')
        return 0
    pct = 100.0 * covered / total
    floor = gate_setting('coverage', 'diff_floor', DEFAULT_DIFF_FLOOR, BANNER)
    if pct < floor:
        print('DIFF COVERAGE GATE -- FAIL', file=sys.stderr)
        print(f'  changed-line coverage {pct:.1f}% (required >= {floor}%)', file=sys.stderr)
        for rel, miss in uncovered:
            preview = ','.join(str(n) for n in miss[:8])
            print(f'    {rel}: uncovered changed lines {preview}', file=sys.stderr)
        print('Merge blocked.', file=sys.stderr)
        return 1
    print(f'DIFF COVERAGE GATE -- PASS (changed-line coverage {pct:.1f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
