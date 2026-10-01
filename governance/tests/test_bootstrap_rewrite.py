"""The inherited bootstrap preserves Talos identity and uses safe administrative setup."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

bootstrap = importlib.import_module('bootstrap_repository')


def test_existing_talos_file_bootstrap_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / 'talos'
    package.mkdir()
    source = package / '__init__.py'
    source.write_bytes((bootstrap.REPO_ROOT / 'talos/__init__.py').read_bytes())
    budgets = tmp_path / '.github/budgets.json'
    budgets.parent.mkdir()
    budgets.write_text(json.dumps({'modules': {'talos/__init__.py': 999}}))
    before = (source.read_bytes(), budgets.read_bytes())
    monkeypatch.setattr(bootstrap, 'REPO_ROOT', tmp_path)
    bootstrap._apply_file_bootstrap('talos', 'talos', 'autonomio')
    bootstrap._apply_file_bootstrap('talos', 'talos', 'autonomio')
    assert before == (source.read_bytes(), budgets.read_bytes())


def test_github_bootstrap_delegates_to_local_nondestructive_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = []
    monkeypatch.setenv('GH_TOKEN', 'test-token')
    monkeypatch.setattr(bootstrap.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    bootstrap._apply_github_bootstrap('talos', 'autonomio')
    assert len(commands) == 1
    assert commands[0][-3:] == ['--repo', 'autonomio/talos', '--apply']
    assert commands[0][1].endswith('scripts/configure_repository.py')
