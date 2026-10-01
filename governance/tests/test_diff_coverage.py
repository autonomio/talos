"""Diff coverage gate: parsing and evaluation are correct on synthetic input."""
from __future__ import annotations

import importlib
import types


def _load() -> types.ModuleType:
    # governance/ is on sys.path via governance/tests/conftest.py.
    return importlib.import_module('check_diff_coverage')


def test_parse_added_lines_tracks_new_file_line_numbers() -> None:
    mod = _load()
    diff = (
        'diff --git a/pkg/m.py b/pkg/m.py\n'
        '--- a/pkg/m.py\n'
        '+++ b/pkg/m.py\n'
        '@@ -0,0 +3,2 @@\n'
        '+x = 1\n'
        '+y = 2\n'
        '@@ -10 +12 @@\n'
        '-old = 1\n'
        '+new = 1\n'
    )
    assert mod.parse_added_lines(diff) == {'pkg/m.py': {3, 4, 12}}


def test_evaluate_flags_uncovered_changed_lines() -> None:
    mod = _load()
    added = {'pkg/m.py': {3, 4, 12}}
    files = {'pkg/m.py': {'executed_lines': [3, 4], 'missing_lines': [12]}}
    total, covered, uncovered = mod.evaluate(added, files)
    assert (total, covered) == (3, 2)
    assert uncovered == [('pkg/m.py', [12])]


def test_evaluate_ignores_non_executable_added_lines() -> None:
    mod = _load()
    # Line 9 is neither executed nor missing (blank/comment) -> not counted.
    added = {'pkg/m.py': {3, 9}}
    files = {'pkg/m.py': {'executed_lines': [3], 'missing_lines': []}}
    total, covered, uncovered = mod.evaluate(added, files)
    assert (total, covered, uncovered) == (1, 1, [])
