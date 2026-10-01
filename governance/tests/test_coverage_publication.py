"""Coverage publication retains the executed report and trusted source binding."""
import hashlib
import json
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
    assert all(step['if'] == "steps.current.outputs.publish == 'true'" for step in job['steps'][1:])
    assert 'CURRENT_SHA' in job['steps'][0]['run']
