"""Reject altered source wheels, incomplete audit graphs and unproved findings."""
import contextlib
import copy
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from tools.security.legacy_audit import dispositions
from tools.security.patches import apply_patch, digest
from tools.security.wheels import MANIFEST, ROOT, security_wheel, verified_contents


def test_exact_patch_rejects_source_context_and_output_drift():
    before, after = b'one\ntwo\n', b'one\nrepaired\n'
    patch = '--- a\n+++ b\n@@ -1,2 +1,2 @@\n one\n-two\n+repaired\n'
    assert apply_patch(before, patch, digest(before), digest(after)) == after
    with pytest.raises(ValueError, match='upstream source hash'):
        apply_patch(b'tampered\n', patch, digest(before), digest(after))
    with pytest.raises(ValueError, match='context mismatch'):
        apply_patch(before, patch.replace('-two', '-foreign'), digest(before), digest(after))
    with pytest.raises(ValueError, match='result hash'):
        apply_patch(before, patch, digest(before), digest(b'foreign output'))
    with pytest.raises(ValueError, match='hunk size'):
        apply_patch(before, patch.replace('-1,2 +1,2', '-1,3 +1,2'), digest(before), digest(after))


def test_upstream_record_cannot_omit_or_duplicate_wheel_members():
    for names in [('package.py', 'package.py'), ('package.py', 'unrecorded.py')]:
        stream = io.BytesIO()
        with pytest.warns(UserWarning) if names[0] == names[1] else contextlib.nullcontext():
            with zipfile.ZipFile(stream, 'w') as archive:
                for name in names:
                    archive.writestr(name, b'fixture source')
                archive.writestr('fixture.dist-info/RECORD', 'fixture.dist-info/RECORD,,\n')
        with pytest.raises(ValueError, match=r'duplicate members|complete wheel'):
            verified_contents(stream.getvalue(), 'fixture.dist-info/')


def test_reviewed_security_wheel_is_reproducible_and_refuses_replacement(tmp_path):
    # Real official input and exact reviewed patches; no manufactured source wheel.
    entry = json.loads(MANIFEST.read_text())[0]
    cache, output = tmp_path / 'cache', tmp_path / 'output'
    first = security_wheel(entry, cache, output)
    assert first['sha256'] == entry['output_sha256']
    assert first['installed_tree_sha256'] == entry['installed_tree_sha256']
    assert security_wheel(entry, cache, output) == first
    artifact = output / first['wheel']
    artifact.write_bytes(b'different existing artifact')
    with pytest.raises(ValueError, match='Refuse to replace'):
        security_wheel(entry, cache, output)
    assert artifact.read_bytes() == b'different existing artifact'
    (cache / entry['wheel']).write_bytes(b'altered official input')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        security_wheel(entry, cache, output)


def test_builder_rejects_source_patch_manifest_drift(tmp_path, monkeypatch):
    entry = copy.deepcopy(json.loads(MANIFEST.read_text())[0])
    patch = entry['patches'][0]
    patch_root = tmp_path / 'patches'
    patch_root.mkdir()
    for declared in entry['patches']:
        (patch_root / declared['patch']).write_bytes((ROOT / 'patches' / declared['patch']).read_bytes())
    path = patch_root / patch['patch']
    path.write_text(path.read_text().replace('+', '-', 1))
    monkeypatch.setattr('tools.security.wheels.ROOT', tmp_path)
    with pytest.raises(ValueError, match='unified file diff'):
        security_wheel(entry, tmp_path / 'cache', tmp_path / 'output')


@pytest.mark.parametrize('mutation', ['missing-package', 'duplicate-package', 'wrong-version', 'missing-findings', 'unknown-advisory'])
def test_legacy_audit_refuses_incomplete_or_unproved_findings(mutation):
    identities = {'dependencies': [{'name': 'requests', 'audit_version': '2.33.0'}]}
    report = {'dependencies': [{'name': 'requests', 'version': '2.33.0', 'vulns': []}]}
    if mutation == 'missing-package':
        report['dependencies'] = []
    elif mutation == 'duplicate-package':
        report['dependencies'] *= 2
    elif mutation == 'wrong-version':
        report['dependencies'][0]['version'] = '2.0.0'
    elif mutation == 'missing-findings':
        del report['dependencies'][0]['vulns']
    else:
        report['dependencies'][0]['vulns'] = [{'id': 'GHSA-unreviewed-fixture'}]
    with pytest.raises(ValueError, match=r'complete installed dependency graph|identity or findings missing|Unrepaired legacy advisory'):
        dispositions(report, identities)


def test_auditor_execution_error_cannot_accept_a_stale_report(tmp_path, monkeypatch):
    from tools.security.legacy_audit import main

    report = tmp_path / 'audit.json'
    report.write_text('{"dependencies": []}')
    output = tmp_path / 'dispositions.json'
    monkeypatch.setattr(sys, 'argv', ['legacy_audit', '--execute', '--requirements', str(tmp_path / 'requirements'),
                        '--audit', str(report), '--identities', str(tmp_path / 'identities'), '--output', str(output)])
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: subprocess.CompletedProcess(args, 2))
    with pytest.raises(ValueError, match='auditor failed: exit 2'):
        main()
    assert not output.exists()


def test_owned_dependency_monitor_is_periodic_bounded_and_not_a_pr_burst():
    workflow = Path(__file__).resolve().parents[2] / '.github/workflows/dependency_monitor.yml'
    data = yaml.safe_load(workflow.read_text())
    triggers = data.get('on', data.get(True))
    assert set(triggers) == {'schedule', 'workflow_dispatch'}
    assert triggers['schedule'] == [{'cron': '20 2 * * 4'}]
    assert data['permissions'] == {'contents': 'read'}
    assert list(data['jobs']) == ['monitor_dependencies']
    job = data['jobs']['monitor_dependencies']
    assert job['timeout-minutes'] == 15
    commands = '\n'.join(step.get('run', '') for step in job['steps'])
    assert 'pip install --require-hashes -r requirements/ci/legacy-3.11.txt' in commands
    assert 'tools.security.legacy_audit --execute' in commands
    assert 'npm --prefix docs-site run security:audit' in commands
    assert not any(step.get('continue-on-error') for step in job['steps'])
    retention = job['steps'][-1]
    assert retention['if'] == 'always()'
    assert 'legacy-advisory-dispositions.json' in retention['with']['path']
