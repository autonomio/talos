"""The first-adoption boundary expires when protected history carries governance."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import _common
import pytest


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _commit(root: Path) -> str:
    _git(root, 'add', '.')
    _git(root, '-c', 'user.name=Governance test', '-c', 'user.email=test@example.invalid',
         'commit', '-qm', 'test: record boundary fixture')
    return _git(root, 'rev-parse', 'HEAD')


def _history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    _git(tmp_path, 'init', '-q')
    (tmp_path / 'source.py').write_text('"""Boundary fixture."""\n')
    base = _commit(tmp_path)
    (tmp_path / 'source.py').write_text('"""Completed pre-adoption fixture."""\n')
    anchor = _commit(tmp_path)
    config = tmp_path / 'governance.yml'
    config.write_text(json.dumps({'adoption': {'baseline_commit': anchor}}))
    monkeypatch.setattr(_common, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(_common, 'GOVERNANCE_CONFIG', config)
    return base, anchor


def test_ungoverned_base_uses_only_an_ancestral_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, anchor = _history(tmp_path, monkeypatch)
    assert _common.comparison_ref(base) == anchor


def test_governed_base_ignores_even_a_changed_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _ = _history(tmp_path, monkeypatch)
    governed = _commit(tmp_path)
    _common.GOVERNANCE_CONFIG.write_text(json.dumps({'adoption': {'baseline_commit': 'invalid'}}))
    assert _common.comparison_ref(governed) == governed


def test_unreachable_base_never_becomes_initial_adoption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _history(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _common.comparison_ref('origin/missing')


def test_boundary_cannot_move_before_the_protected_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, anchor = _history(tmp_path, monkeypatch)
    _common.GOVERNANCE_CONFIG.write_text(json.dumps({'adoption': {'baseline_commit': base}}))
    with pytest.raises(SystemExit):
        _common.comparison_ref(anchor)


def test_boundary_must_be_ancestor_of_the_judged_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _ = _history(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        _common.comparison_ref(base, base)
