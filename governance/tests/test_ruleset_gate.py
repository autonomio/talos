from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RULESET_GATE = REPO_ROOT / 'governance/ruleset_gate.py'
SNAPSHOT = REPO_ROOT / '.github/rulesets/master.json'
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
