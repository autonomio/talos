"""Tests for the dependency-vulnerability gate's pure decision logic."""
from __future__ import annotations

import datetime
import importlib

import pytest

# governance/ is on sys.path via governance/tests/conftest.py.
gate = importlib.import_module('check_dependency_vulnerabilities')

_TODAY = datetime.date(2026, 6, 14)
_AUDITED = [
    {
        'name': 'jinja2',
        'version': '2.11.0',
        'vulns': [
            {'id': 'PYSEC-2021-66', 'fix_versions': ['2.11.3']},
            {'id': 'GHSA-x', 'fix_versions': []},
        ],
    }
]


def test_evaluate_reports_every_unexcepted_vuln() -> None:
    findings = gate.evaluate(_AUDITED, set())
    assert len(findings) == 2
    assert any('PYSEC-2021-66' in item for item in findings)


def test_evaluate_suppresses_an_excepted_id() -> None:
    findings = gate.evaluate(_AUDITED, {'PYSEC-2021-66'})
    assert len(findings) == 1
    assert all('PYSEC-2021-66' not in item for item in findings)


def test_evaluate_reports_missing_fix_as_none_published() -> None:
    findings = gate.evaluate(_AUDITED, {'PYSEC-2021-66'})
    assert 'none published' in findings[0]


def test_active_exceptions_honors_expiry() -> None:
    active = gate.active_exceptions('[{"id":"A","reason":"tracked","expiry":"2099-01-01"}]', _TODAY)
    assert active == {'A'}
    expired = gate.active_exceptions('[{"id":"A","reason":"tracked","expiry":"2020-01-01"}]', _TODAY)
    assert expired == set()


def test_active_exceptions_requires_a_reason() -> None:
    assert gate.active_exceptions('[{"id":"A","reason":"  ","expiry":"2099-01-01"}]', _TODAY) == set()


def test_active_exceptions_empty_text_is_empty() -> None:
    assert gate.active_exceptions('', _TODAY) == set()


def test_active_exceptions_rejects_malformed_entry() -> None:
    with pytest.raises(SystemExit):
        gate.active_exceptions('[{"id":"A"}]', _TODAY)


def test_active_exceptions_rejects_bad_expiry() -> None:
    with pytest.raises(SystemExit):
        gate.active_exceptions('[{"id":"A","reason":"x","expiry":"not-a-date"}]', _TODAY)
