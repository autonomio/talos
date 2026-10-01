#!/usr/bin/env python3
"""Test fallback gate: tests must not use try/except (use pytest.raises).

Scans every module under the test directories. Test *infrastructure* is not
a test -- a runner, a fixture factory, a profiling plugin -- and a
`try`/`finally` there restores state rather than swallowing an assertion.
Paths listed under `gates.test_fallbacks.excludes` in `governance.yml`
are skipped; the default excludes nothing, so this repository is unchanged.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from _common import (
    REPO_ROOT,
    exit_if_disabled,
    fail_setup,
    find_python_files,
    gate_config,
    resolve_paths,
)

BANNER = 'TEST FALLBACK GATE'


def _excludes() -> list[str]:
    """Configured infrastructure paths, or none. Fails closed on a value
    that is not a list of strings: a gate cannot skip what it cannot read."""
    raw = gate_config('test_fallbacks', BANNER).get('excludes', [])
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        fail_setup(BANNER, f'test_fallbacks.excludes must be a list of strings, got {raw!r}')
    return raw


def find_try_statements(source: str) -> list[int]:
    # A test proves exception behavior with `pytest.raises`, never with a
    # try/except that can swallow a failing assertion. Flag every try (and
    # the 3.11+ try/except* form) by line number.
    tree = ast.parse(source)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Try, ast.TryStar))
    ]


def main() -> int:
    exit_if_disabled('test_fallbacks', BANNER)
    # Read before the scan so an unreadable config blocks even when no test
    # directory exists.
    excludes = _excludes()
    violations: list[tuple[Path, int]] = []
    for test_dir in [
        *resolve_paths('test_paths', BANNER),
        *resolve_paths('gate_test_paths', BANNER),
    ]:
        if not test_dir.exists():
            continue
        for path in find_python_files(test_dir, [*excludes, '__pycache__']):
            for lineno in find_try_statements(path.read_text(encoding='utf-8')):
                violations.append((path.relative_to(REPO_ROOT), lineno))
    if violations:
        print('TEST FALLBACK GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for rel, lineno in violations:
            print(
                f'  - {rel}:{lineno}: try/except in a test; use pytest.raises instead',
                file=sys.stderr,
            )
        print('', file=sys.stderr)
        print(f'{len(violations)} violation(s). Merge blocked.', file=sys.stderr)
        return 1
    print('TEST FALLBACK GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
