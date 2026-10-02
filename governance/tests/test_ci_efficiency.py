"""CI reduces runner demand without weakening source, coverage or packaging checks."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / '.github/workflows'


def load(name: str) -> dict[str, object]:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def test_required_test_and_lint_names_share_same_run_successful_coverage() -> None:
    workflow = load('pr_checks_lint.yml')
    jobs = workflow['jobs']
    assert jobs['pr_checks_tests']['name'] == 'pr_checks_tests'
    lint = jobs['pr_checks_lint']
    assert lint['name'] == 'pr_checks_lint'
    assert lint['needs'] == 'pr_checks_tests'
    assert lint['if'] == '${{ always() && !cancelled() }}'
    producer = jobs['pr_checks_tests']['steps']
    upload = next(step for step in producer if step.get('id') == 'coverage')
    assert 'if' not in upload  # A failed suite cannot publish partial gating evidence.
    assert upload['with']['if-no-files-found'] == 'error'
    runs = '\n'.join(step.get('run', '') for step in producer)
    assert '-m pytest governance/tests/ -q' in runs
    assert 'coverage_evidence.py --write' in runs
    assert producer.index(upload) > next(index for index, step in enumerate(producer)
                                         if step.get('name') == 'Enforce the suite runtime budget')
    steps = lint['steps']
    assert steps[0]['run'].startswith("test '${{ needs.pr_checks_tests.result }}' = success")
    download = next(step for step in steps if 'download-artifact@' in step.get('uses', ''))
    assert download['with'] == {'artifact-ids': '${{ needs.pr_checks_tests.outputs.coverage_artifact }}'}
    assert 'continue-on-error' not in download
    assert not any('-m pytest' in step.get('run', '') for step in steps)
    publisher = jobs['publish_coverage_comment']
    assert publisher['permissions']['pull-requests'] == 'write'
    assert not any('checkout@' in step.get('uses', '') for step in publisher['steps'])


def test_heavy_source_checks_cancel_superseded_heads_and_have_total_timeouts() -> None:
    for name in ('ci.yml', 'pr_checks_lint.yml', 'pr_checks_packaging.yml',
                 'pr_checks_codeql.yml', 'pr_checks_honesty.yml', 'pr_checks_ruleset.yml'):
        workflow = load(name)
        assert workflow['concurrency']['cancel-in-progress'] is True, name
        assert 'github.workflow' in workflow['concurrency']['group'], name
        assert 'github.sha' not in workflow['concurrency']['group'], name
        for job in workflow['jobs'].values():
            assert 0 < job['timeout-minutes'] <= 45, name
    assert not (WORKFLOWS / 'pr_checks_tests.yml').exists()
    assert set(load('pr_checks_packaging.yml')['jobs']) == {'pr_checks_packaging'}


def test_metadata_edits_recheck_markers_without_rerunning_scientific_suites() -> None:
    lint = load('pr_checks_lint.yml')
    events = lint.get('on', lint.get(True))
    assert 'edited' not in events['pull_request']['types']
    version = load('pr_checks_version.yml')
    assert 'edited' in version.get('on', version.get(True))['pull_request']['types']
    commands = '\n'.join(step.get('run', '') for step in version['jobs']['pr_checks_version']['steps'])
    assert 'check_budget_ratchet.py' in commands
    assert 'check_coverage_ratchet.py' in commands
    assert 'check_test_runtime.py --check-ratchet-only' in commands
    assert '-m pytest' not in commands


def test_dependency_bursts_are_grouped_and_scheduled_independently() -> None:
    updates = yaml.safe_load((ROOT / '.github/dependabot.yml').read_text())['updates']
    assert len(updates) == 3
    slots = set()
    for update in updates:
        assert update['open-pull-requests-limit'] == 1
        schedule = update['schedule']
        assert schedule['timezone'] == 'Europe/Helsinki'
        slots.add((schedule['day'], schedule['time']))
        for kind in ('version-updates', 'security-updates'):
            assert update['groups'][kind] == {'applies-to': kind, 'patterns': ['*']}
        assert 'ignore' not in update
        assert 'target-branch' not in update
    assert len(slots) == 3
