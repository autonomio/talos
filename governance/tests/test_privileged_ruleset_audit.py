from __future__ import annotations

import contextlib
import importlib
import io
import json
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RULESET_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_ruleset.yml'
AUDIT_WORKFLOW = REPO_ROOT / '.github/workflows/audit_master_ruleset.yml'
SNAPSHOT = REPO_ROOT / '.github/rulesets/master.json'
FIXTURES = REPO_ROOT / 'governance/tests/fixtures/github'


def _load_audit_module() -> types.ModuleType:
    # governance/ is on sys.path via governance/tests/conftest.py.
    return importlib.import_module('privileged_ruleset_audit')


def _run_audit_fixture(live_fixture: str, tmp_path: Path) -> tuple[int, str, str]:
    module = _load_audit_module()
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = module.run_audit(
            ruleset_file=str(SNAPSHOT),
            repo=None,
            ruleset_id=1,
            output_dir=str(tmp_path),
            live_json=str(FIXTURES / live_fixture),
        )
    return code, stdout.getvalue(), stderr.getvalue()


def test_privileged_ruleset_audit_accepts_matching_live_payload_with_bypass_actors(
    tmp_path: Path,
) -> None:
    code, stdout, stderr = _run_audit_fixture('ruleset_live_target.json', tmp_path)

    assert code == 0
    assert 'PRIVILEGED RULESET AUDIT -- PASS' in stdout
    assert stderr == ''
    assert not (tmp_path / 'live_ruleset.json').exists()


def test_privileged_ruleset_audit_fails_on_bypass_actor_drift(tmp_path: Path) -> None:
    code, _, stderr = _run_audit_fixture('ruleset_live_with_bypass_actor.json', tmp_path)

    assert code == 1
    assert 'privileged_ruleset_audit: ruleset drift detected' in stderr


def test_privileged_ruleset_audit_writes_live_payload_snapshot_on_failure(
    tmp_path: Path,
) -> None:
    code, _, stderr = _run_audit_fixture('ruleset_live_with_bypass_actor.json', tmp_path)
    snapshot = tmp_path / 'live_ruleset.json'

    assert code == 1
    assert snapshot.exists()
    assert json.loads(snapshot.read_text(encoding='utf-8')) == json.loads(
        (FIXTURES / 'ruleset_live_with_bypass_actor.json').read_text(encoding='utf-8')
    )
    assert str(snapshot) in stderr


def test_privileged_ruleset_audit_fails_loud_when_live_payload_omits_bypass_actors(
    tmp_path: Path,
) -> None:
    code, _, stderr = _run_audit_fixture(
        'ruleset_live_target_without_bypass_actors.json',
        tmp_path,
    )
    snapshot = tmp_path / 'live_ruleset.json'

    assert code == 2
    assert (
        "privileged live ruleset missing required observable field(s): ['bypass_actors']"
        in stderr
    )
    assert snapshot.exists()
    assert json.loads(snapshot.read_text(encoding='utf-8')) == json.loads(
        (FIXTURES / 'ruleset_live_target_without_bypass_actors.json').read_text(
            encoding='utf-8'
        )
    )


def test_audit_main_ruleset_workflow_contract() -> None:
    workflow = AUDIT_WORKFLOW.read_text(encoding='utf-8')

    assert 'name: audit_master_ruleset' in workflow
    assert 'push:' in workflow
    assert 'branches: [master]' in workflow
    assert 'workflow_dispatch:' in workflow
    assert "if: github.ref == 'refs/heads/master'" in workflow
    assert 'pull_request:' not in workflow
    assert 'RULESET_AUDIT_TOKEN' in workflow
    assert 'governance/privileged_ruleset_audit.py' in workflow
    assert '--ruleset-file .github/rulesets/master.json' in workflow
    assert '--repo "${{ github.repository }}"' in workflow
    # The template resolves the ruleset id from the RULESET_ID repository
    # variable (set by bootstrap), never a hardcoded id.
    assert 'vars.RULESET_ID' in workflow
    assert '--ruleset-id "$RULESET_ID"' in workflow
    assert '--output-dir .ruleset-audit' in workflow
    assert 'actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a  # v7.0.1' in workflow
    assert 'failure()' in workflow
    assert '.ruleset-audit' in workflow


def test_pr_checks_ruleset_runs_privileged_audit_contract() -> None:
    workflow = RULESET_WORKFLOW.read_text(encoding='utf-8')

    assert 'governance/tests/test_privileged_ruleset_audit.py' in workflow
    assert 'continue-on-error' not in workflow
