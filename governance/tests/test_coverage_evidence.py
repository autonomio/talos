"""Shared coverage must reject stale runs, source changes and artifact tampering."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from governance import coverage_evidence


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'source.py').write_text('value = 1\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test',
                    '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'test: source'], check=True)
    source = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()
    locks = tmp_path / 'requirements/ci'
    locks.mkdir(parents=True)
    for name in ('dev-env.txt', 'runtime-env.txt', 'build-tools.txt'):
        (locks / name).write_text('locked\n')
    (tmp_path / 'coverage.json').write_text('{"totals": {"covered_lines": 1}}\n')
    monkeypatch.setattr(coverage_evidence, 'ROOT', tmp_path)
    monkeypatch.setattr(coverage_evidence, 'RECEIPT', tmp_path / 'coverage-evidence.json')
    monkeypatch.setenv('GITHUB_SHA', source)
    monkeypatch.setenv('GITHUB_RUN_ID', '123')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '1')
    monkeypatch.setenv('TEST_PRODUCER_ATTEMPT', '1')
    coverage_evidence.RECEIPT.write_text(json.dumps(coverage_evidence.binding()))
    return tmp_path


def test_successful_same_source_and_run_evidence(evidence: Path) -> None:
    coverage_evidence.verify()


@pytest.mark.parametrize('name', ['GITHUB_RUN_ID', 'TEST_PRODUCER_ATTEMPT', 'GITHUB_SHA'])
def test_foreign_run_attempt_or_source_is_rejected(
    evidence: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    monkeypatch.setenv(name, 'changed')
    with pytest.raises(ValueError):
        coverage_evidence.verify()


@pytest.mark.parametrize('path', ['coverage.json', 'requirements/ci/runtime-env.txt'])
def test_changed_artifact_or_lock_is_rejected(evidence: Path, path: str) -> None:
    (evidence / path).write_text('changed')
    with pytest.raises(ValueError):
        coverage_evidence.verify()


def test_missing_producer_receipt_is_rejected(evidence: Path) -> None:
    coverage_evidence.RECEIPT.unlink()
    with pytest.raises(FileNotFoundError):
        coverage_evidence.verify()


def test_consumer_retry_reuses_the_exact_successful_producer_attempt(
    evidence: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '2')
    coverage_evidence.verify()
