"""Documentation reuse and deadlines preserve executable evidence and terminate children."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.verify_documentation import execute_component

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('development_evidence', ROOT / 'tools/verification/development_evidence.py')
assert SPEC is not None and SPEC.loader is not None
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


@pytest.fixture
def development_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source = tmp_path / 'test_model.py'
    source.write_text('original model\n')
    proof = tmp_path / 'proof.json'
    import hashlib
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'test_model.py'], check=True)
    monkeypatch.delenv('TALOS_DOC_SOURCE_ROOT', raising=False)
    proof.write_text(json.dumps({
        'command': EVIDENCE.COMMAND, 'returncode': 0, 'run_id': '1',
        'source': 'source', 'run_attempt': '1',
        'files': {'test_model.py': hashlib.sha256(source.read_bytes()).hexdigest()},
        'runtime': {'dependencies': 'locked'},
    }))
    monkeypatch.setenv('GITHUB_RUN_ID', '1')
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT', '1')
    monkeypatch.setenv('GITHUB_SHA', 'source')
    monkeypatch.setattr(EVIDENCE, 'environment', lambda root, python: {'dependencies': 'locked'})
    return tmp_path, proof


def test_matching_successful_development_command_is_reusable(development_proof: tuple[Path, Path]) -> None:
    root, proof = development_proof
    assert EVIDENCE.verify(root, sys.executable, proof)['returncode'] == 0


@pytest.mark.parametrize('field,value', [('command', ['pytest']), ('returncode', 1), ('run_id', '2'),
                                          ('runtime', {'dependencies': 'changed'}),
                                          ('source', 'changed'), ('run_attempt', '2'), ('files', {})])
def test_other_command_failed_run_or_environment_is_rejected(
    development_proof: tuple[Path, Path], field: str, value: object,
) -> None:
    root, proof = development_proof
    data = json.loads(proof.read_text())
    data[field] = value
    proof.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        EVIDENCE.verify(root, sys.executable, proof)


def test_changed_model_source_invalidates_development_reuse(development_proof: tuple[Path, Path]) -> None:
    root, proof = development_proof
    (root / 'test_model.py').write_text('changed model\n')
    with pytest.raises(ValueError, match='source changed'):
        EVIDENCE.verify(root, sys.executable, proof)


def test_new_tracked_source_invalidates_development_reuse(development_proof: tuple[Path, Path]) -> None:
    root, proof = development_proof
    (root / 'new_model.py').write_text('new source\n')
    subprocess.run(['git', '-C', str(root), 'add', 'new_model.py'], check=True)
    with pytest.raises(ValueError, match='tracked inventory'):
        EVIDENCE.verify(root, sys.executable, proof)


def test_untracked_fixture_test_cannot_reuse_smaller_suite(development_proof: tuple[Path, Path]) -> None:
    root, proof = development_proof
    (root / 'tests').mkdir()
    (root / 'tests/test_new.py').write_text('def test_new():\n    assert True\n')
    with pytest.raises(ValueError, match='fixture code inventory'):
        EVIDENCE.verify(root, sys.executable, proof)


def test_untracked_root_model_cannot_reuse_smaller_inputs(development_proof: tuple[Path, Path]) -> None:
    root, proof = development_proof
    (root / 'new_model.py').write_text('new source\n')
    with pytest.raises(ValueError, match='fixture code inventory'):
        EVIDENCE.verify(root, sys.executable, proof)


def test_expired_shared_deadline_does_not_start_queued_component(tmp_path: Path) -> None:
    marker = tmp_path / 'started'
    with pytest.raises(TimeoutError, match='before component launch'):
        execute_component([sys.executable, '-c', f'open({str(marker)!r}, "w").close()'],
                          tmp_path, os.environ.copy(), tmp_path / 'log', time.monotonic() - 1)
    assert not marker.exists()


def test_deadline_kills_executor_and_descendant_process(tmp_path: Path) -> None:
    pid = tmp_path / 'child.pid'
    script = ('import subprocess,sys,time,pathlib; '
              'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
              f'pathlib.Path({str(pid)!r}).write_text(str(child.pid)); time.sleep(60)')
    started = time.monotonic()
    with pytest.raises(TimeoutError, match='during component execution'):
        execute_component([sys.executable, '-c', script], tmp_path, os.environ.copy(),
                          tmp_path / 'log', started + 1)
    assert time.monotonic() - started < 5
    child = pid.read_text()
    state = subprocess.run(['ps', '-o', 'stat=', '-p', child], text=True,
                           capture_output=True, check=False)
    assert state.returncode != 0 or state.stdout.strip().startswith('Z'), state.stdout


def test_failing_executor_cannot_become_successful_receipt(tmp_path: Path) -> None:
    receipt = execute_component([sys.executable, '-c', 'raise SystemExit(7)'], tmp_path,
                                os.environ.copy(), tmp_path / 'log', time.monotonic() + 5)
    assert receipt['returncode'] == 7
