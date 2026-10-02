"""Execute release boundaries without publishing or manufacturing signed evidence."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from _common import REPO_ROOT

WORKFLOW = REPO_ROOT / '.github/workflows/deploy.yml'


def _jobs():
    return yaml.safe_load(WORKFLOW.read_text())['jobs']


def _run_step(job, name, root, environment):
    step = next(step for step in _jobs()[job]['steps'] if step.get('name') == name)
    script = root / 'workflow-step.sh'
    script.write_text(step['run'])
    return subprocess.run(['bash', str(script)], cwd=root, env=environment,
                          capture_output=True, text=True, check=False)


def _git(root, *args):
    return subprocess.run(['git', *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.fixture
def release_repo(tmp_path):
    origin = tmp_path / 'origin.git'
    origin.mkdir()
    _git(origin, 'init', '--bare', '--initial-branch=master')
    root = tmp_path / 'checkout'
    root.mkdir()
    _git(root, 'init', '--initial-branch=master')
    _git(root, 'config', 'user.name', 'Release fixture')
    _git(root, 'config', 'user.email', 'release-fixture@example.test')
    (root / 'source.txt').write_text('reviewed source\n')
    _git(root, 'add', 'source.txt')
    _git(root, 'commit', '-m', 'chore: prepare release fixture')
    _git(root, 'remote', 'add', 'origin', str(origin))
    _git(root, 'push', 'origin', 'master')
    _git(root, 'tag', 'v2.0.2')
    environment = {'PATH': os.environ['PATH'], 'GIT_CONFIG_GLOBAL': os.devnull,
                   'GIT_CONFIG_NOSYSTEM': '1'}
    environment.update(GH_TOKEN='fixture-token', RELEASE_TAG='v2.0.2',
                       GITHUB_SHA=_git(root, 'rev-parse', 'HEAD'),
                       GITHUB_OUTPUT=str(tmp_path / 'job-output'))
    return root, environment


def test_release_source_accepts_only_integrated_matching_commit(release_repo):
    root, environment = release_repo
    result = _run_step('build', 'Validate reviewed release commit', root, environment)
    assert result.returncode == 0, result.stderr
    assert Path(environment['GITHUB_OUTPUT']).read_text() == f"source_sha={environment['GITHUB_SHA']}\n"


def test_release_source_rejects_option_like_tag_before_fetch(release_repo):
    root, environment = release_repo
    environment['RELEASE_TAG'] = '--upload-packages'
    result = _run_step('build', 'Validate reviewed release commit', root, environment)
    assert result.returncode != 0
    assert 'vMAJOR.MINOR.PATCH' in result.stdout
    assert not Path(environment['GITHUB_OUTPUT']).exists()


@pytest.mark.parametrize('mutation', ['unreviewed', 'wrong-workflow-source', 'wrong-checkout'])
def test_release_source_rejects_unreviewed_or_misattributed_commit(release_repo, mutation):
    root, environment = release_repo
    old_sha = environment['GITHUB_SHA']
    if mutation == 'unreviewed':
        _git(root, 'checkout', '-b', 'unreviewed')
    (root / 'source.txt').write_text('new source\n')
    _git(root, 'commit', '-am', 'chore: modify release fixture')
    new_sha = _git(root, 'rev-parse', 'HEAD')
    if mutation == 'unreviewed':
        _git(root, 'tag', '-f', 'v2.0.2')
        environment['GITHUB_SHA'] = new_sha
    else:
        _git(root, 'push', 'origin', 'master')
        if mutation == 'wrong-workflow-source':
            environment['GITHUB_SHA'] = new_sha
            _git(root, 'checkout', '--detach', old_sha)
    result = _run_step('build', 'Validate reviewed release commit', root, environment)
    assert result.returncode != 0
    assert not Path(environment['GITHUB_OUTPUT']).exists()
    expected = 'integrated into protected master' if mutation == 'unreviewed' else 'same commit'
    assert expected in result.stdout


@pytest.fixture
def release_assets(tmp_path):
    dist = tmp_path / 'dist'
    evidence = tmp_path / 'release-evidence'
    tools = tmp_path / 'bin'
    for directory in [dist, evidence, tools]:
        directory.mkdir()
    files = {'talos-2.0.2-py3-none-any.whl': b'wheel fixture',
             'talos-2.0.2.tar.gz': b'source fixture'}
    for name, content in files.items():
        (dist / name).write_bytes(content)
    (evidence / 'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256(content).hexdigest()}  {name}\n' for name, content in files.items()))
    # Verification is mocked; these bytes are never represented as a valid signature.
    (evidence / 'talos-v2.0.2.sigstore.json').write_text('mock verification input\n')
    gh = tools / 'gh'
    gh.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['GH_LOG']).open('a') as output:
    output.write(json.dumps(args) + '\\n')
if args[:2] == ['attestation', 'verify']:
    sys.exit(1 if os.environ.get('FAIL_VERIFICATION') == 'true' else 0)
elif args[:2] == ['release', 'view']:
    if '--jq' not in args:
        print(json.dumps({'isDraft': os.environ.get('DRAFT') == 'true',
                          'tagName': os.environ.get('METADATA_TAG', 'v2.0.2')}))
    elif 'assets' in args:
        print(os.environ.get('EXISTING_ASSETS', ''))
    else:
        print(os.environ.get('RELEASE_STATE', 'v2.0.2'))
elif args[:2] != ['release', 'upload']:
    raise SystemExit('unexpected fixture command')
''')
    gh.chmod(0o755)
    environment = {'PATH': os.environ['PATH'], 'GIT_CONFIG_GLOBAL': os.devnull,
                   'GIT_CONFIG_NOSYSTEM': '1'}
    environment.update(PATH=str(tools) + os.pathsep + str(Path(sys.executable).parent)
                       + os.pathsep + environment['PATH'],
                       GH_TOKEN='fixture-token', RELEASE_TAG='v2.0.2',
                       GITHUB_REPOSITORY='autonomio/talos', GITHUB_REF='refs/heads/master',
                       SOURCE_SHA='a' * 40, GH_LOG=str(tmp_path / 'gh-log'))
    return tmp_path, environment


def _gh_calls(environment):
    path = Path(environment['GH_LOG'])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_release_attachment_verifies_every_subject_before_immutable_upload(release_assets):
    root, environment = release_assets
    result = _run_step('release_assets', 'Verify and attach immutable release assets', root, environment)
    assert result.returncode == 0, result.stderr
    calls = _gh_calls(environment)
    verified = [call for call in calls if call[:2] == ['attestation', 'verify']]
    assert len(verified) == 3
    assert {call[2] for call in verified} == {
        'dist/talos-2.0.2-py3-none-any.whl', 'dist/talos-2.0.2.tar.gz', 'release-evidence/SHA256SUMS'}
    for call in verified:
        for flag, expected in [('repo', 'autonomio/talos'), ('source-digest', 'a' * 40),
                               ('source-ref', 'refs/heads/master'),
                               ('signer-workflow', 'autonomio/talos/.github/workflows/deploy.yml'),
                               ('bundle', 'release-evidence/talos-v2.0.2.sigstore.json')]:
            assert call[call.index('--' + flag) + 1] == expected
        assert '--deny-self-hosted-runners' in call
    assert calls[-1][:2] == ['release', 'upload']
    assert '--clobber' not in calls[-1]
    assert 'release-evidence/talos-v2.0.2.sigstore.json' in calls[-1]


@pytest.mark.parametrize('failure', ['checksum', 'signature', 'existing-asset', 'draft'])
def test_release_attachment_fails_before_any_upload(release_assets, failure):
    root, environment = release_assets
    if failure == 'checksum':
        (root / 'dist/talos-2.0.2.tar.gz').write_bytes(b'tampered fixture')
    elif failure == 'signature':
        environment['FAIL_VERIFICATION'] = 'true'
    elif failure == 'existing-asset':
        environment['EXISTING_ASSETS'] = 'unrelated.txt\ntalos-v2.0.2.sigstore.json'
    else:
        environment['RELEASE_STATE'] = 'draft'
    result = _run_step('release_assets', 'Verify and attach immutable release assets', root, environment)
    assert result.returncode != 0
    assert not any(call[:2] == ['release', 'upload'] for call in _gh_calls(environment))


@pytest.mark.parametrize('draft,tag,version', [
    (False, 'v2.0.2', '2.0.2'), (True, 'v2.0.2', '2.0.2'),
    (False, 'v9.0.0', '2.0.2'), (False, 'v2.0.2', '9.0.0'),
])
def test_release_metadata_rejects_drafts_and_wrong_versions(release_assets, draft, tag, version):
    root, environment = release_assets
    (root / 'scripts').mkdir()
    shutil.copyfile(REPO_ROOT / 'scripts/create_release.py', root / 'scripts/create_release.py')
    (root / 'talos').mkdir()
    (root / 'talos/__init__.py').write_text(f'__version__ = "{version}"\n')
    (root / 'pyproject.toml').write_text('[project]\ndynamic = ["version"]\n'
                                        '[tool.hatch.version]\npath = "talos/__init__.py"\n')
    environment.update(DRAFT='true' if draft else 'false', METADATA_TAG=tag,
                       RELEASE_METADATA=str(root / 'release-metadata.json'))
    result = _run_step('build', 'Validate published tag and package version', root, environment)
    assert (result.returncode == 0) == (not draft and tag == 'v2.0.2' and version == '2.0.2'), result.stderr


def test_release_privilege_and_pypi_enablement_are_separate_contracts():
    jobs = _jobs()
    assert 'PYPI_PUBLISH_ENABLED' not in jobs['build']['if']
    assert jobs['publish']['if'] == "vars.PYPI_PUBLISH_ENABLED == 'true'"
    assert jobs['publish']['needs'] == ['build', 'release_assets']
    assert jobs['publish']['environment'] == 'pypi'
    assert jobs['publish']['permissions'] == {'id-token': 'write'}
    assert jobs['build']['permissions']['contents'] == 'read'
    assert jobs['release_assets']['permissions'] == {'contents': 'write'}
    assert not any('checkout@' in step.get('uses', '') for step in jobs['release_assets']['steps'])
    uploads = [step for step in jobs['build']['steps'] if 'upload-artifact@' in step.get('uses', '')]
    artifacts = {step['with']['name']: step['with']['path'].splitlines() for step in uploads}
    assert artifacts['release-dist'] == ['dist/*.whl', 'dist/*.tar.gz']
    assert artifacts['release-evidence'] == ['release-evidence/SHA256SUMS', 'release-evidence/*.sigstore.json']
    attest = next(step for step in jobs['build']['steps'] if step.get('id') == 'attestation')
    assert attest['with']['subject-path'].splitlines() == ['dist/*.whl', 'dist/*.tar.gz', 'release-evidence/SHA256SUMS']
    bundle = next(step for step in jobs['build']['steps'] if step.get('name') == 'Retain the authentic Sigstore bundle')
    assert bundle['env']['BUNDLE_PATH'] == '${{ steps.attestation.outputs.bundle-path }}'
    pypi_guard = next(step for step in jobs['build']['steps'] if step.get('name') == 'Guard against a reused PyPI version')
    assert pypi_guard['if'] == "vars.PYPI_PUBLISH_ENABLED == 'true'"
