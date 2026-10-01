#!/usr/bin/env python3
"""File size balance gate: largest <= max_ratio x median.

The bound is per-repository: a package of many small single-purpose
modules and a package of a few large ones are different shapes, and a
ratio that is right for one is arbitrary for the other. Configured under
`gates.file_size_balance.max_ratio` in `governance.yml`; the default
reproduces the previously hardcoded 16.00.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Final

from _common import (
    REPO_ROOT,
    exit_if_disabled,
    fail_setup,
    gate_config,
    gate_setting,
    resolve_package_dir,
)

# A deliberately lenient bootstrap default: 16x lets a young package carry a
# framework-boundary module or two that must stay physically together, while
# still blocking an accidental "one huge file" architecture. The gate is
# dormant below MIN_FILES_FOR_GATE files, so it only bites once a package has
# real structure; tighten the ratio as it grows and the largest-file
# exception stops reflecting a real boundary.
BANNER: Final[str] = 'FILE SIZE BALANCE GATE'
DEFAULT_MAX_RATIO: Final[float] = 16.00
DEFAULT_MIN_FILES: Final[int] = 3


def count_lines(path: Path) -> int:
    # Total physical line count, blank lines included (file-size balance
    # cares about actual file size on disk, not logical SLOC).
    return len(path.read_text(encoding='utf-8').splitlines())


def _max_ratio() -> float:
    """The configured bound, or the default when unset.

    Fails closed on a value that is not a positive number: a gate cannot
    check a shape against a bound it cannot parse.
    """
    raw = gate_config('file_size_balance', BANNER).get('max_ratio', DEFAULT_MAX_RATIO)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        fail_setup(BANNER, f'file_size_balance.max_ratio must be a positive number, got {raw!r}')
    return float(raw)


def main() -> int:
    exit_if_disabled('file_size_balance', BANNER)
    source_dir = resolve_package_dir(BANNER)
    # Resolved before the dormancy checks below: a malformed or missing
    # config must block on every path, not only the one that reaches the
    # ratio comparison. Otherwise a repository with too few files to balance
    # would pass while its configuration was unreadable.
    max_ratio = _max_ratio()
    min_files = gate_setting('file_size_balance', 'min_files', DEFAULT_MIN_FILES, BANNER)
    sized: list[tuple[Path, int]] = [
        (p, count_lines(p)) for p in sorted(source_dir.rglob('*.py'))
    ]
    if len(sized) < min_files:
        # Too few files to have a size imbalance. The scan target exists
        # (resolved and checked above); this is a legitimately small
        # package, not a misconfiguration.
        print(
            f'FILE SIZE BALANCE GATE -- PASS '
            f'(only {len(sized)} source file(s), need >= {min_files} to balance)'
        )
        return 0
    # Exclude zero-line files from the median: a package with many
    # empty __init__.py files would otherwise produce median=0 and
    # ratio=inf for the smallest real file. Empty files don't have a
    # meaningful size to balance against.
    nonzero_sizes = [s for _, s in sized if s > 0]
    if len(nonzero_sizes) < min_files:
        print(
            f'FILE SIZE BALANCE GATE -- PASS '
            f'(only {len(nonzero_sizes)} non-empty source file(s) to balance)'
        )
        return 0
    largest_path, largest_size = max(sized, key=lambda item: item[1])
    median = statistics.median(nonzero_sizes)
    ratio = largest_size / median
    if ratio > max_ratio:
        print('FILE SIZE BALANCE GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        rel = largest_path.relative_to(REPO_ROOT)
        print(f'  largest file:    {rel} ({largest_size} lines)', file=sys.stderr)
        print(f'  median file size: {int(median)} lines', file=sys.stderr)
        print(
            f'  ratio:            {ratio:.2f} (max allowed: {max_ratio:.2f})',
            file=sys.stderr,
        )
        print('', file=sys.stderr)
        print('Merge blocked.', file=sys.stderr)
        return 1
    print(
        f'FILE SIZE BALANCE GATE -- PASS '
        f'(largest={largest_size}, median={int(median)}, ratio={ratio:.2f})'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
