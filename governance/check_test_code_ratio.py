#!/usr/bin/env python3
"""Test/code SLOC ratio gate."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from _common import (
    exit_if_disabled,
    gate_setting,
    resolve_package_dir,
    resolve_paths,
    significant_lines,
)

BANNER = 'TEST/CODE RATIO GATE'
DEFAULT_MIN_RATIO: Final[float] = 0.60
DEFAULT_MAX_RATIO: Final[float] = 2.00
DEFAULT_MIN_SOURCE_SLOC: Final[int] = 50


def count_py_sloc(root: Path) -> int:
    if root.is_file():
        return significant_lines(root)
    if not root.is_dir():
        return 0
    return sum(significant_lines(path) for path in sorted(root.rglob('*.py')))


def main() -> int:
    exit_if_disabled('test_code_ratio', BANNER)
    source_dir = resolve_package_dir('TEST/CODE RATIO GATE')
    source = count_py_sloc(source_dir)
    test = sum(count_py_sloc(path) for path in resolve_paths('test_paths', BANNER))
    min_ratio = gate_setting('test_code_ratio', 'min', DEFAULT_MIN_RATIO, BANNER)
    max_ratio = gate_setting('test_code_ratio', 'max', DEFAULT_MAX_RATIO, BANNER)
    min_sloc = gate_setting('test_code_ratio', 'min_source_sloc', DEFAULT_MIN_SOURCE_SLOC, BANNER)
    if source < min_sloc:
        # The scan target exists (resolved and checked above); there is
        # just not yet enough source for a ratio to be meaningful.
        print(f'TEST/CODE RATIO GATE -- PASS (source {source} SLOC < {min_sloc}, ratio not yet meaningful)')
        return 0
    ratio = test / source if source > 0 else 0.0
    if ratio < min_ratio or ratio > max_ratio:
        print('TEST/CODE RATIO GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        print(f'  source SLOC: {source} ({source_dir.name}/)', file=sys.stderr)
        print(f'  test   SLOC: {test} (tests/)', file=sys.stderr)
        print(
            f'  ratio:       {ratio:.2f} '
            f'(required: [{min_ratio:.2f}, {max_ratio:.2f}])',
            file=sys.stderr,
        )
        print('', file=sys.stderr)
        print('Merge blocked.', file=sys.stderr)
        return 1
    print(f'TEST/CODE RATIO GATE -- PASS (source={source}, test={test}, ratio={ratio:.2f})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
