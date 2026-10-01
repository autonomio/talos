"""Unit tests for the coverage floor (FLOOR + TRACK) and the coverage ratchet."""
from __future__ import annotations

import importlib
import types

import pytest


def _mod(name: str) -> types.ModuleType:
    # governance/ is on sys.path via governance/tests/conftest.py.
    return importlib.import_module(name)


floor = _mod('check_coverage_floor')
ratchet = _mod('check_coverage_ratchet')


# --- FLOOR gate: coverage-key extraction --------------------------------

def test_line_pct_prefers_new_key() -> None:
    assert floor._line_pct({'percent_statements_covered': 87.5, 'percent_covered': 1.0}) == 87.5


def test_line_pct_falls_back_to_blended_key() -> None:
    assert floor._line_pct({'percent_covered': 64.0}) == 64.0


def test_branch_pct_is_vacuously_complete_without_branches() -> None:
    assert floor._branch_pct({'num_branches': 0, 'percent_branches_covered': 0.0}) == 100.0


def test_branch_pct_uses_branch_key_when_branches_exist() -> None:
    assert floor._branch_pct({'num_branches': 12, 'percent_branches_covered': 71.0}) == 71.0


# --- FLOOR gate: TRACK rule ---------------------------------------------

def test_track_dormant_below_min_statements() -> None:
    # 100% actual, 50% floor, but only 10 statements -> no demand to bank.
    assert floor._track_violation('line', 100.0, 50, 10, floor.DEFAULT_MIN_STATEMENTS, floor.DEFAULT_TRACK_SLACK) is None


def test_track_fires_when_floor_lags_actual() -> None:
    msg = floor._track_violation('line', 90.0, 50, 200, floor.DEFAULT_MIN_STATEMENTS, floor.DEFAULT_TRACK_SLACK)
    assert msg is not None
    assert '>= 88%' in msg  # floor(90) - TRACK_SLACK(2)


def test_track_silent_within_slack() -> None:
    # actual 89, floor 88 -> lag 1 <= slack 2 -> satisfied.
    assert floor._track_violation('line', 89.0, 88, 200, floor.DEFAULT_MIN_STATEMENTS, floor.DEFAULT_TRACK_SLACK) is None


def test_track_branch_dormant_below_min_branches() -> None:
    assert floor._track_violation('branch', 100.0, 45, 5, floor.DEFAULT_MIN_BRANCHES, floor.DEFAULT_TRACK_SLACK) is None


# --- RATCHET gate: floor parsing ----------------------------------------

def test_parse_floor_reads_line_and_branch() -> None:
    assert ratchet._parse_floor('{\"coverage\": {\"line\": 80, \"branch\": 70}}') == {'line': 80, 'branch': 70}


def test_parse_floor_empty_text_is_empty_dict() -> None:
    assert ratchet._parse_floor('   ') == {}


def test_parse_floor_rejects_out_of_range() -> None:
    with pytest.raises(SystemExit):
        ratchet._parse_floor('{\"coverage\": {\"line\": 150, \"branch\": 70}}')


def test_parse_floor_rejects_non_int() -> None:
    with pytest.raises(SystemExit):
        ratchet._parse_floor('{\"coverage\": {\"line\": 80.5, \"branch\": 70}}')


def test_parse_floor_rejects_bool() -> None:
    with pytest.raises(SystemExit):
        ratchet._parse_floor('{\"coverage\": {\"line\": true, \"branch\": 70}}')


# --- RATCHET gate: lowering detection -----------------------------------

def test_unmarked_lowering_of_both_fields_flags_both() -> None:
    base = {'line': 80, 'branch': 70}
    head = {'line': 50, 'branch': 45}
    assert ratchet.check(base, head, '') == [('line', 80, 50), ('branch', 70, 45)]


def test_per_field_marker_clears_only_that_field() -> None:
    base = {'line': 80, 'branch': 70}
    head = {'line': 50, 'branch': 45}
    body = '[coverage-lower: line: dropped a heavily tested module]\n'
    assert ratchet.check(base, head, body) == [('branch', 70, 45)]


def test_raising_the_floor_needs_no_marker() -> None:
    base = {'line': 50, 'branch': 45}
    head = {'line': 80, 'branch': 70}
    assert ratchet.check(base, head, '') == []


def test_marker_requires_a_reason() -> None:
    # A field with no reason text does not match the marker.
    assert ratchet.LOWER_MARKER_RE.search('[coverage-lower: line: ]') is None
    assert ratchet.LOWER_MARKER_RE.search('[coverage-lower: line: real reason]') is not None
