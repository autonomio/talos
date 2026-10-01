#!/usr/bin/env python3
"""Test-suite runtime gate: the suite completes inside its recorded budget.

CLAUDE.md's stance says a command that runs repeatedly must be profiled and
its profile reported, not merely tolerated. This gate is that stance made
mechanical: the suite writes a runtime profile, and the profile is checked
against a committed ceiling.

The ceiling is per-repository, in `.github/budgets.json`. The template's
own suite is trivial, so its ceiling is loose scaffolding a derived repository
recalibrates against its own observed spread -- the shape of the budget file
records the observation window so a later reader can tell a measured ceiling
from a guessed one.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

from _common import REPO_ROOT, exit_if_disabled, fail_setup, gate_setting

BANNER: Final[str] = 'TEST RUNTIME GATE'
BUDGET_PATH: Final[Path] = REPO_ROOT / '.github' / 'budgets.json'
BUDGET_SECTION: Final[str] = 'runtime'
DEFAULT_SLOWEST_TESTS_LIMIT: Final[int] = 10
RAISE_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r'^\[runtime-raise:\s*(?P<reason>.*?\S)\s*\]\s*$',
    re.MULTILINE,
)


def load_json(path: Path, what: str) -> dict[str, Any]:
    """Read one JSON object, failing closed when it is absent or malformed."""
    if not path.is_file():
        fail_setup(BANNER, f'missing {what}: {path}')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        fail_setup(BANNER, f'cannot read {what} at {path}: {exc}')
    if not isinstance(data, dict):
        fail_setup(BANNER, f'{what} at {path} is not a JSON object')
    return data


def base_ceiling(base_ref: str | None, base_file: str | None) -> float | None:
    """The ceiling recorded on the base ref, or None when there is none.

    None means the base ref carries no runtime budget at all -- the commit
    introducing it. A base ref that has one but cannot be read is a setup
    failure rather than a None, because a silently absent base ceiling would
    make every raise look like a first commit.
    """
    if base_ref is not None:
        # `git show REF:path` fails both when the file is absent at REF and
        # when REF itself is unreachable. Those must not be conflated: the
        # first is the commit introducing the budget, the second is a base ref
        # that was never fetched -- and treating that as "no base ceiling"
        # would silently disable the ratchet exactly when it cannot be checked.
        if subprocess.run(
            ['git', 'rev-parse', '--verify', '--quiet', f'{base_ref}^{{commit}}'],
            check=False, capture_output=True, text=True,
        ).returncode != 0:
            fail_setup(BANNER, f'base ref {base_ref!r} is unreachable; fetch it before the gate runs')
        result = subprocess.run(
            ['git', 'show', f'{base_ref}:.github/budgets.json'],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        text = result.stdout
    elif base_file is not None:
        # Same rule as the branch above: an unreadable base is a setup failure,
        # not an absent ceiling. Falling through to None here would skip the
        # ratchet on a typo'd path -- the loudest possible way to be silent.
        if not Path(base_file).is_file():
            fail_setup(BANNER, f'base file {base_file!r} is not a regular file')
        text = Path(base_file).read_text(encoding='utf-8')
    else:
        return None
    if not text.strip():
        return None
    try:
        section = json.loads(text).get(BUDGET_SECTION, {})
    except json.JSONDecodeError as exc:
        fail_setup(BANNER, f'cannot parse base budgets.json: {exc}')
    if not isinstance(section, dict):
        fail_setup(BANNER, 'base budgets.json runtime section is not an object')
    value = section.get('max_total_seconds')
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        fail_setup(BANNER, f'base max_total_seconds must be positive, got {value!r}')
    return float(value)


def raise_is_declared(pr_body: str) -> bool:
    """Whether the PR body carries a `[runtime-raise: <reason>]` line.

    The ceiling is a ceiling, so *raising* it is the loosening that needs a
    reason. Lowering needs no marker -- that is the ratchet working.
    """
    return RAISE_MARKER_RE.search(pr_body) is not None


def slowest(profile: dict[str, Any], limit: int) -> list[tuple[str, float]]:
    """Return the slowest tests, longest first, capped at limit."""
    raw = profile.get('tests', [])
    if not isinstance(raw, list):
        fail_setup(BANNER, 'profile .tests must be a list')
    rows: list[tuple[str, float]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get('name', '<unnamed>'))
        duration = entry.get('duration', 0.0)
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            continue
        rows.append((name, float(duration)))
    return sorted(rows, key=lambda row: -row[1])[:limit]


def main() -> int:
    exit_if_disabled('runtime_budget', BANNER)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, help='runtime profile JSON')
    parser.add_argument('--enforce', action='store_true',
                        help='exit non-zero when the suite exceeds its ceiling')
    parser.add_argument('--base-ref', help='protected base ref to compare the ceiling against')
    parser.add_argument('--base-file', help='path to the base budgets.json (local/test mode)')
    parser.add_argument('--pr-body-file', help='file holding the PR body, for the raise marker')
    args = parser.parse_args()

    budget = load_json(BUDGET_PATH, 'runtime budget').get(BUDGET_SECTION, {})
    profile = load_json(Path(args.profile), 'runtime profile')

    ceiling = budget.get('max_total_seconds')
    if isinstance(ceiling, bool) or not isinstance(ceiling, (int, float)) or ceiling <= 0:
        fail_setup(BANNER, f'max_total_seconds must be a positive number, got {ceiling!r}')
    limit = gate_setting(
        'runtime_budget', 'slowest_tests_limit', DEFAULT_SLOWEST_TESTS_LIMIT, BANNER
    )

    base = base_ceiling(args.base_ref, args.base_file)
    if base is not None and float(ceiling) > base:
        body_path = args.pr_body_file
        pr_body = (
            Path(body_path).read_text(encoding='utf-8')
            if body_path is not None and Path(body_path).is_file()
            else ''
        )
        if not raise_is_declared(pr_body):
            print(f'{BANNER} -- FAIL', file=sys.stderr)
            print('', file=sys.stderr)
            print(
                f'  raised without marker: max_total_seconds '
                f'(base={base:g}, head={float(ceiling):g}, +{float(ceiling) - base:g})',
                file=sys.stderr,
            )
            print('', file=sys.stderr)
            print(
                '  PR body must contain `[runtime-raise: <reason>]` on its own line. '
                'A ceiling raised by the PR it gates is not a ceiling.',
                file=sys.stderr,
            )
            print('Merge blocked.', file=sys.stderr)
            return 1

    total = profile.get('total_seconds')
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        fail_setup(BANNER, f'profile .total_seconds must be a number, got {total!r}')

    print(f'suite runtime: {float(total):.2f}s (ceiling {float(ceiling):.2f}s)')
    rows = slowest(profile, limit)
    if rows:
        print(f'slowest {len(rows)}:')
        for name, duration in rows:
            print(f'  {duration:7.2f}s  {name}')

    if float(total) > float(ceiling):
        if not args.enforce:
            print(f'{BANNER} -- PASS (over ceiling, not enforcing)')
            return 0
        print(f'{BANNER} -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        print(
            f'  suite took {float(total):.2f}s against a {float(ceiling):.2f}s ceiling.',
            file=sys.stderr,
        )
        print(
            '  Profile the slowest tests above and make them faster; raising the '
            'ceiling is the last resort, not the first.',
            file=sys.stderr,
        )
        print('', file=sys.stderr)
        print('Merge blocked.', file=sys.stderr)
        return 1

    print(f'{BANNER} -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
