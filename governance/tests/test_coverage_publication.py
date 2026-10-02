"""Coverage publication retains the executed report and trusted source binding."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools.coverage_badge import publish

ROOT = Path(__file__).resolve().parents[2]


def test_badge_preserves_executed_report_and_source_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr('tools.coverage_badge.check_statement_coverage',
                        lambda report, source: (4, 5))
    report = tmp_path / 'report.json'
    source = b'{"files":{"talos/one.py":{}},"totals":{}}\n'
    report.write_bytes(source)
    commit = 'a' * 40
    run_url = 'https://github.com/autonomio/talos/actions/runs/123'
    result = publish(report, tmp_path / 'public', commit, run_url)
    assert result['statement_percent'] == 80.0
    assert result['source_commit'] == commit
    assert result['report_sha256'] == hashlib.sha256(source).hexdigest()
    assert (tmp_path / 'public/coverage.json').read_bytes() == source
    assert '80.00%' in (tmp_path / 'public/coverage.svg').read_text()
    assert commit in (tmp_path / 'public/coverage.html').read_text()
    assert run_url in (tmp_path / 'public/coverage.html').read_text()
    with pytest.raises(ValueError, match='full lowercase'):
        publish(report, tmp_path / 'invalid', 'master', run_url)
    with pytest.raises(ValueError, match='Talos GitHub'):
        publish(report, tmp_path / 'invalid', commit, 'https://example.org/123')


def test_failed_complete_statement_gate_creates_no_badge(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'files': {}, 'totals': {}}))
    output = tmp_path / 'public'
    with pytest.raises(ValueError, match='every package module'):
        publish(report, output, 'b' * 40, 'https://github.com/autonomio/talos/actions/runs/123')
    assert not output.exists()


def test_pages_publication_never_executes_pull_request_source():
    workflow = yaml.safe_load((ROOT / '.github/workflows/docs_pages.yml').read_text())
    job = workflow['jobs']['publish']
    assert "workflow_run.event == 'push'" in job['if']
    assert "workflow_run.conclusion == 'success'" in job['if']
    assert "workflow_run.head_repository.full_name == github.repository" in job['if']
    assert all('checkout@' not in step.get('uses', '') for step in job['steps'])
    download = next(step for step in job['steps'] if 'download-artifact@' in step.get('uses', ''))
    assert download['with']['run-id'] == '${{ github.event.workflow_run.id }}'
    assert download['with']['name'] == 'documentation-site'
    assert "github.repository == 'autonomio/talos'" in job['if']
    assert all(step['if'] == "steps.current.outputs.publish == 'true'" for step in job['steps'][1:-1])
    assert job['steps'][-1]['if'] == "steps.deployable.outputs.publish == 'true'"
    assert 'CURRENT_SHA' in job['steps'][0]['run']


def test_pages_pending_publications_are_isolated_by_source_commit() -> None:
    workflow = yaml.safe_load((ROOT / '.github/workflows/docs_pages.yml').read_text())
    group = workflow['concurrency']['group']
    assert group == 'talos-pages-${{ github.event.workflow_run.head_sha }}'
    assert workflow['concurrency']['cancel-in-progress'] is False
    steps = workflow['jobs']['publish']['steps']
    assert steps[-2]['id'] == 'deployable'
    assert 'upload-pages-artifact@' in steps[-3]['uses']
    assert 'deploy-pages@' in steps[-1]['uses']


@pytest.mark.parametrize('step_id', ['current', 'deployable'])
@pytest.mark.parametrize('master_sha', ['a' * 40, 'b' * 40])
def test_pages_guards_check_live_master_before_using_artifacts(
    tmp_path: Path, step_id: str, master_sha: str,
) -> None:
    workflow = yaml.safe_load((ROOT / '.github/workflows/docs_pages.yml').read_text())
    step = next(item for item in workflow['jobs']['publish']['steps'] if item.get('id') == step_id)
    gh = tmp_path / 'gh'
    gh.write_text('#!/bin/sh\nset -eu\n'
                  'test "$*" = "api repos/autonomio/talos/git/ref/heads/master --jq .object.sha"\n'
                  'printf "%s\\n" "$TEST_MASTER_SHA"\n')
    gh.chmod(0o755)
    output = tmp_path / 'output'
    summary = tmp_path / 'summary'
    environment = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
                   'SOURCE_SHA': 'a' * 40, 'TEST_MASTER_SHA': master_sha,
                   'GITHUB_REPOSITORY': 'autonomio/talos',
                   'GITHUB_OUTPUT': str(output), 'GITHUB_STEP_SUMMARY': str(summary)}
    subprocess.run(['bash', '-c', step['run']], env=environment, check=True, capture_output=True, text=True)
    assert output.read_text() == ('publish=true\n' if master_sha == 'a' * 40 else 'publish=false\n')
    if master_sha != 'a' * 40:
        assert 'Skip superseded documentation build' in summary.read_text()


@pytest.mark.parametrize('repository', ['researcher/talos', 'autonomio/another-project'])
def test_fork_ci_skips_canonical_badge_generation_without_writing_files(
    tmp_path: Path, repository: str,
) -> None:
    output = tmp_path / 'public'
    result = subprocess.run(
        [sys.executable, '-m', 'tools.coverage_badge', str(tmp_path / 'missing-report.json'),
         '--output', str(output), '--commit', 'a' * 40,
         '--run-url', f'https://github.com/{repository}/actions/runs/123'],
        cwd=ROOT, env={**os.environ, 'GITHUB_REPOSITORY': repository},
        capture_output=True, text=True, check=True,
    )
    assert 'Skip Talos coverage publication outside autonomio/talos' in result.stderr
    assert result.stdout == ''
    assert not output.exists()


def test_canonical_ci_still_refuses_foreign_run_provenance(tmp_path: Path) -> None:
    output = tmp_path / 'public'
    result = subprocess.run(
        [sys.executable, '-m', 'tools.coverage_badge', str(tmp_path / 'missing-report.json'),
         '--output', str(output), '--commit', 'a' * 40,
         '--run-url', 'https://github.com/researcher/talos/actions/runs/123'],
        cwd=ROOT, env={**os.environ, 'GITHUB_REPOSITORY': 'autonomio/talos'},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert 'coverage must link to a Talos GitHub Actions run' in result.stderr
    assert not output.exists()


def test_fork_publication_skip_leaves_complete_statement_gate_unconditional() -> None:
    workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    jobs = workflow['jobs'].values()
    steps = next(job['steps'] for job in jobs if any(
        'coverage_badge' in step.get('run', '') for step in job['steps']))
    publication_index = next(index for index, step in enumerate(steps) if 'coverage_badge' in step.get('run', ''))
    gate = steps[publication_index - 1]
    assert 'tools/check_statement_coverage.py verification-output/coverage.json' in gate['run']
    assert 'if' not in gate


@pytest.mark.parametrize('step_id', ['current', 'deployable'])
def test_failed_master_lookup_cannot_authorize_publication(tmp_path: Path, step_id: str) -> None:
    workflow = yaml.safe_load((ROOT / '.github/workflows/docs_pages.yml').read_text())
    step = next(item for item in workflow['jobs']['publish']['steps'] if item.get('id') == step_id)
    gh = tmp_path / 'gh'
    gh.write_text('#!/bin/sh\nexit 1\n')
    gh.chmod(0o755)
    output = tmp_path / 'output'
    result = subprocess.run(
        ['bash', '-c', step['run']], capture_output=True, text=True, check=False,
        env={**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
             'SOURCE_SHA': 'a' * 40, 'GITHUB_REPOSITORY': 'autonomio/talos',
             'GITHUB_OUTPUT': str(output), 'GITHUB_STEP_SUMMARY': str(tmp_path / 'summary')},
    )
    assert result.returncode == 1
    assert not output.exists()
