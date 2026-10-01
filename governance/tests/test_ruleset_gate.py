from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RULESET_GATE = REPO_ROOT / 'governance/ruleset_gate.py'
SNAPSHOT = REPO_ROOT / '.github/rulesets/master.json'
CODEQL_PRESENT = 'PR Checks CodeQL (python)' in (REPO_ROOT / 'CLAUDE.md').read_text(encoding='utf-8')
FIXTURES = REPO_ROOT / 'governance/tests/fixtures/github'


def run_gate(live_fixture: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(RULESET_GATE),
            '--live-json',
            str(FIXTURES / live_fixture),
            '--ruleset-file',
            str(SNAPSHOT),
            '--ruleset-id',
            '5406599',
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_matches_target_snapshot() -> None:
    result = run_gate('ruleset_live_target.json')
    assert result.returncode == 0, result.stderr
    assert 'RULESET GATE -- PASS' in result.stdout


def test_missing_bypass_actors_is_allowed_for_repo_hosted_reads() -> None:
    result = run_gate('ruleset_live_target_without_bypass_actors.json')
    assert result.returncode == 0, result.stderr
    assert 'comparing observable subset only' in result.stderr
    assert 'RULESET GATE -- PASS' in result.stdout


def test_bypass_actor_drift_is_failure() -> None:
    result = run_gate('ruleset_live_with_bypass_actor.json')
    assert result.returncode == 1
    assert 'ruleset drift detected' in result.stderr


def test_unexpected_top_level_field_in_live_is_drift() -> None:
    result = run_gate('ruleset_live_unexpected_field.json')
    assert result.returncode == 1
    assert 'unexpected live ruleset field(s)' in result.stderr


def test_ignored_live_fields_match_named_set() -> None:
    namespace: dict[str, object] = {'__name__': 'ruleset_gate'}
    exec(RULESET_GATE.read_text(encoding='utf-8'), namespace)
    assert namespace['IGNORED_LIVE_FIELDS'] == frozenset({
        '_links',
        'created_at',
        'current_user_can_bypass',
        'id',
        'node_id',
        'source',
        'source_type',
        'updated_at',
    })


@pytest.mark.skipif(not CODEQL_PRESENT, reason='CodeQL explicitly removed from this repository')
@pytest.mark.parametrize('security_threshold', [None, 'none', 'high_or_higher', 'medium_or_higher'])
def test_missing_or_weakened_codeql_security_rule_is_drift(
    security_threshold: str | None, tmp_path: Path,
) -> None:
    payload = json.loads((FIXTURES / 'ruleset_live_target.json').read_text(encoding='utf-8'))
    rule = next(rule for rule in payload['rules'] if rule['type'] == 'code_scanning')
    if security_threshold is None:
        payload['rules'].remove(rule)
    else:
        rule['parameters']['code_scanning_tools'][0]['security_alerts_threshold'] = security_threshold
    live_path = tmp_path / 'live_ruleset.json'
    live_path.write_text(json.dumps(payload), encoding='utf-8')

    result = run_gate(str(live_path))

    assert result.returncode == 1
    assert 'ruleset drift detected' in result.stderr
