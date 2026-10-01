"""A budget-raise marker must state a reason to unlock a raise.

The marker exists to record *why* a calibrated budget moved. A marker that
parses with a blank reason unlocks the raise while recording nothing, which is
worse than no marker at all: the gate reports PASS and the PR looks compliant.

`check_budget_ratchet` was the one gate whose reason group could be satisfied
by whitespace. `(?P<reason>.+?)` backtracks onto the space that `\\s*` would
otherwise consume, so `[budget-raise: a/b.py: ]` matched with reason `' '`.
The two sibling markers already spell it `(?P<reason>.*?\\S)`, which cannot end
on whitespace. This aligns the third.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _mod(name: str) -> types.ModuleType:
    # governance/ is on sys.path via governance/tests/conftest.py.
    return importlib.import_module(name)


ratchet = _mod('check_budget_ratchet')


def test_blank_reason_is_rejected() -> None:
    """The exact form that used to pass: a single space where a reason belongs."""
    assert ratchet.RAISE_MARKER_RE.search('[budget-raise: a/b.py: ]') is None


def test_whitespace_only_reason_is_rejected() -> None:
    """Padding the blank out further must not help either."""
    assert ratchet.RAISE_MARKER_RE.search('[budget-raise: a/b.py:    ]') is None
    assert ratchet.RAISE_MARKER_RE.search('[budget-raise: a/b.py:\t]') is None


def test_stated_reason_is_accepted() -> None:
    """The marker still does its job when a reason is actually given."""
    match = ratchet.RAISE_MARKER_RE.search('[budget-raise: a/b.py: the module took on X]')
    assert match is not None
    assert match.group('path').strip() == 'a/b.py'
    assert match.group('reason') == 'the module took on X'


def test_reason_may_contain_a_colon() -> None:
    """Real reasons quote numbers, so the reason cannot stop at a colon.

    The `path` group stops at the first colon by design; everything after it
    belongs to the reason, colons included.
    """
    match = ratchet.RAISE_MARKER_RE.search(
        '[budget-raise: governance/g.py: took the ratchet: 100 -> 165 lines]'
    )
    assert match is not None
    assert match.group('reason') == 'took the ratchet: 100 -> 165 lines'


def test_trailing_whitespace_in_a_real_reason_is_trimmed() -> None:
    """A reason padded on the right still parses, without the padding."""
    match = ratchet.RAISE_MARKER_RE.search('[budget-raise: a/b.py: a real reason   ]')
    assert match is not None
    assert match.group('reason') == 'a real reason'


def test_markers_are_found_inside_a_realistic_pr_body() -> None:
    """Markers are matched per line inside prose, and blanks are skipped."""
    body = (
        'Some PR description.\n'
        '\n'
        '[budget-raise: a/good.py: a stated reason]\n'
        '[budget-raise: a/blank.py: ]\n'
        '\n'
        'More text.\n'
    )
    found = {
        m.group('path').strip() for m in ratchet.RAISE_MARKER_RE.finditer(body)
    }
    assert found == {'a/good.py'}


def test_every_marker_rejects_a_blank_reason() -> None:
    """All three ratchet markers must agree on this.

    Pinned across the three rather than on this gate alone: the bug was that
    one of them was an outlier, and a test covering only the gate it was found
    in would not notice the next outlier.
    """
    coverage = _mod('check_coverage_ratchet')
    runtime = _mod('check_test_runtime')
    assert ratchet.RAISE_MARKER_RE.search('[budget-raise: a/b.py: ]') is None
    assert coverage.LOWER_MARKER_RE.search('[coverage-lower: line: ]') is None
    assert runtime.RAISE_MARKER_RE.search('[runtime-raise: ]') is None


def test_gate_blocks_a_raise_carrying_a_blank_marker(tmp_path: Path) -> None:
    """End to end: the gate itself refuses the raise, not just the regex.

    Driving the real gate matters here -- the regex could be correct while the
    call site ignored it.
    """
    (tmp_path / '.github').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'governance').mkdir(parents=True, exist_ok=True)
    (tmp_path / '.github' / 'budgets.json').write_text(
        json.dumps({'modules': {'pkg/mod.py': 200}}), encoding='utf-8',
    )
    base = tmp_path / 'base.json'
    base.write_text(json.dumps({'modules': {'pkg/mod.py': 100}}), encoding='utf-8')
    body = tmp_path / 'body.txt'
    body.write_text('[budget-raise: pkg/mod.py: ]\n', encoding='utf-8')
    for module in ('_common.py', 'check_budget_ratchet.py'):
        (tmp_path / 'governance' / module).write_text(
            (REPO_ROOT / 'governance' / module).read_text(encoding='utf-8'), encoding='utf-8',
        )
    result = subprocess.run(
        [
            sys.executable, str(tmp_path / 'governance' / 'check_budget_ratchet.py'),
            '--base-file', str(base), '--pr-body-file', str(body),
        ],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'raised without marker: pkg/mod.py' in result.stderr
