"""The runtime ceiling cannot be raised by the PR it gates.

`max_total_seconds` is a ceiling, so *raising* it is the loosening that needs
a stated reason; lowering it is the ratchet working and needs no marker. This
mirrors `check_budget_ratchet` (`[budget-raise: ...]`) and
`check_coverage_ratchet` (`[coverage-lower: ...]`) rather than inventing a
third convention.

Every case here drives the real gate as a subprocess. The gap this closes was
found when a gate's compare path raised `NameError` in CI: nothing called it,
so nothing could catch it. Asserting on the gate's own exit code and stderr is
the only form that would have.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / 'governance' / 'check_test_runtime.py'


def _run(
    tmp_path: Path,
    head_ceiling: int,
    base_ceiling: int | None,
    pr_body: str,
) -> subprocess.CompletedProcess[str]:
    """Run the gate over a throwaway repo with the given base and head."""
    (tmp_path / '.github').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'governance').mkdir(parents=True, exist_ok=True)
    (tmp_path / '.github' / 'budgets.json').write_text(
        json.dumps({'runtime': {'max_total_seconds': head_ceiling}}), encoding='utf-8',
    )
    base_file = tmp_path / 'base.json'
    base_file.write_text(
        json.dumps({} if base_ceiling is None else {'runtime': {'max_total_seconds': base_ceiling}}),
        encoding='utf-8',
    )
    body_file = tmp_path / 'body.txt'
    body_file.write_text(pr_body, encoding='utf-8')
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'total_seconds': 1.0, 'tests': []}), encoding='utf-8')

    for module in ('_common.py', 'check_test_runtime.py'):
        (tmp_path / 'governance' / module).write_text(
            (REPO_ROOT / 'governance' / module).read_text(encoding='utf-8'), encoding='utf-8',
        )
    return subprocess.run(
        [
            sys.executable, str(tmp_path / 'governance' / 'check_test_runtime.py'),
            '--profile', str(profile),
            '--base-file', str(base_file),
            '--pr-body-file', str(body_file),
        ],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )


def test_raise_without_marker_fails(tmp_path: Path) -> None:
    """A raised ceiling with no stated reason blocks the merge."""
    result = _run(tmp_path, head_ceiling=600, base_ceiling=120, pr_body='no marker here')
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'raised without marker: max_total_seconds' in result.stderr
    assert 'base=120' in result.stderr
    assert 'head=600' in result.stderr


def test_raise_with_marker_passes(tmp_path: Path) -> None:
    """A raised ceiling carrying a reason is allowed through."""
    result = _run(
        tmp_path,
        head_ceiling=600,
        base_ceiling=120,
        pr_body='[runtime-raise: the suite now runs the browser accessibility pass]',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TEST RUNTIME GATE -- PASS' in result.stdout


def test_lowering_needs_no_marker(tmp_path: Path) -> None:
    """Tightening the ceiling is the ratchet working, not a loosening."""
    result = _run(tmp_path, head_ceiling=60, base_ceiling=120, pr_body='')
    assert result.returncode == 0, result.stdout + result.stderr


def test_unchanged_ceiling_needs_no_marker(tmp_path: Path) -> None:
    """An untouched ceiling is not a raise."""
    result = _run(tmp_path, head_ceiling=120, base_ceiling=120, pr_body='')
    assert result.returncode == 0, result.stdout + result.stderr


def test_absent_base_ceiling_is_the_introducing_commit(tmp_path: Path) -> None:
    """A base ref with no runtime budget is the commit introducing it.

    This is the one case that must pass without a marker, and it is also the
    one an attacker would reach for: deleting the base budget to make a raise
    look like a first commit. That is why `check_test_runtime` reads the base
    from the protected ref rather than from anything the PR can write.
    """
    result = _run(tmp_path, head_ceiling=600, base_ceiling=None, pr_body='')
    assert result.returncode == 0, result.stdout + result.stderr


def test_marker_must_carry_a_reason(tmp_path: Path) -> None:
    """An empty marker is not a reason, so it does not unlock the raise."""
    result = _run(tmp_path, head_ceiling=600, base_ceiling=120, pr_body='[runtime-raise: ]')
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'raised without marker' in result.stderr


def test_unreachable_base_ref_blocks(tmp_path: Path) -> None:
    """An unfetched base ref is a setup failure, not an absent ceiling.

    `git show REF:path` fails the same way for "no file at REF" and "no such
    REF". Conflating them would silently disable the ratchet in exactly the
    case where it cannot be evaluated -- a CI job that forgot to fetch the
    base ref would report PASS.
    """
    (tmp_path / '.github').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'governance').mkdir(parents=True, exist_ok=True)
    (tmp_path / '.github' / 'budgets.json').write_text(
        json.dumps({'runtime': {'max_total_seconds': 9999}}), encoding='utf-8',
    )
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'total_seconds': 1.0, 'tests': []}), encoding='utf-8')
    for module in ('_common.py', 'check_test_runtime.py'):
        (tmp_path / 'governance' / module).write_text(
            (REPO_ROOT / 'governance' / module).read_text(encoding='utf-8'), encoding='utf-8',
        )
    result = subprocess.run(
        [
            sys.executable, str(tmp_path / 'governance' / 'check_test_runtime.py'),
            '--profile', str(profile), '--base-ref', 'origin/no-such-ref-exists',
        ],
        capture_output=True, text=True, check=False, cwd=REPO_ROOT,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'unreachable' in result.stderr


def test_missing_base_file_blocks(tmp_path: Path) -> None:
    """A `--base-file` path that is not a regular file is a setup failure.

    The sibling `--base-ref` branch already blocks on an unreachable base.
    This branch fell through to "no base ceiling" instead, so a typo'd path
    skipped the raise check entirely -- the ratchet silently disabled in the
    one case where it could not be evaluated. Distinct from
    `test_absent_base_ceiling_is_the_introducing_commit`, which passes a real
    file whose runtime section is genuinely absent.
    """
    (tmp_path / '.github').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'governance').mkdir(parents=True, exist_ok=True)
    (tmp_path / '.github' / 'budgets.json').write_text(
        json.dumps({'runtime': {'max_total_seconds': 9999}}), encoding='utf-8',
    )
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'total_seconds': 1.0, 'tests': []}), encoding='utf-8')
    for module in ('_common.py', 'check_test_runtime.py'):
        (tmp_path / 'governance' / module).write_text(
            (REPO_ROOT / 'governance' / module).read_text(encoding='utf-8'), encoding='utf-8',
        )
    result = subprocess.run(
        [
            sys.executable, str(tmp_path / 'governance' / 'check_test_runtime.py'),
            '--profile', str(profile),
            '--base-file', str(tmp_path / 'no-such-base.json'),
        ],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'is not a regular file' in result.stderr
