"""Inherited findings are specific evidence, never an exemption for new code."""
from __future__ import annotations

import json
from pathlib import Path

import _quality


def _record() -> dict[str, str]:
    return {'kind': 'ruff', 'path': 'talos/legacy.py', 'message': 'F401: unused import',
            'source_sha256': 'existing-source'}


def test_identical_new_occurrence_is_not_exempted(tmp_path: Path, monkeypatch: object) -> None:
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'schema_version': 1, 'findings': [_record()]}))
    monkeypatch.setattr(_quality, 'BASELINE', baseline)
    assert _quality.new_findings([_record(), _record()]) == [_record()]


def test_changed_source_or_location_is_new_debt(tmp_path: Path, monkeypatch: object) -> None:
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'schema_version': 1, 'findings': [_record()]}))
    monkeypatch.setattr(_quality, 'BASELINE', baseline)
    changed = {**_record(), 'source_sha256': 'edited-source'}
    relocated = {**_record(), 'path': 'talos/new.py'}
    assert _quality.new_findings([changed, relocated]) == [changed, relocated]


def test_removed_debt_needs_no_replacement(tmp_path: Path, monkeypatch: object) -> None:
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'schema_version': 1, 'findings': [_record()]}))
    monkeypatch.setattr(_quality, 'BASELINE', baseline)
    assert _quality.new_findings([]) == []


def test_docstring_evidence_includes_the_docstring_body(tmp_path: Path, monkeypatch: object) -> None:
    path = tmp_path / 'old.py'
    monkeypatch.setattr(_quality, 'REPO_ROOT', tmp_path)
    path.write_text('def method():\n    """Generate old content."""\n')
    old = _quality.finding('docstrings', path, 'forbidden title', 1)
    path.write_text('def method():\n    """Generate changed content."""\n')
    assert _quality.finding('docstrings', path, 'forbidden title', 1) != old


def test_the_judged_branch_cannot_raise_its_own_debt_baseline(
    tmp_path: Path, monkeypatch: object,
) -> None:
    import subprocess

    baseline = tmp_path / 'governance/quality-baseline.json'
    baseline.parent.mkdir()
    baseline.write_text(json.dumps({'schema_version': 1, 'findings': [_record()]}))
    for args in (
        ('init', '-q'), ('add', '.'),
        ('-c', 'user.name=Quality test', '-c', 'user.email=test@example.invalid',
         'commit', '-qm', 'test: record debt baseline'),
    ):
        subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True)
    monkeypatch.setattr(_quality, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(_quality, 'BASELINE', baseline)
    baseline.write_text(json.dumps({'schema_version': 1, 'findings': [_record(), _record()]}))
    assert _quality.baseline_ratchet('HEAD') == [
        'baseline increased: talos/legacy.py: F401: unused import',
    ]
