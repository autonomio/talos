"""Committed identity and configuration checks stop before trusted caller imports."""
import json

import pytest
from click.testing import CliRunner

from talos.cli.main import cli
from talos.yaml.compiler import CompiledSFD
from talos.yaml.errors import ValidationError
from talos.yaml.store import (
    canonical_manifest_id,
    commit_manifest,
    fork_manifest,
    rebuild_index,
    resolve_manifest_uri,
)
from talos.yaml.validator import validate


@pytest.fixture
def committed_project(tmp_path):
    (tmp_path / 'talos.toml').write_text('[store]\nbackup_remote = ""\n')
    marker = tmp_path / 'caller-imported'
    source = tmp_path / 'caller.py'
    source.write_text(
        'from pathlib import Path\n'
        f'Path({str(marker)!r}).write_text("imported")\n'
        "def params(): return {'epochs': [1]}\n"
        'def prep(context, round_params): return context\n'
        "def model(context, round_params): return {'score': round_params['epochs']}\n"
    )
    document = {'schema_version': '1.0', 'metadata': {'name': 'original', 'mode': 'production'},
                'sfd': {'module': source.name, 'params': {'epochs': [1]}},
                'uel': {'search_strategy': {'type': 'grid'}}}
    draft = tmp_path / 'source.yaml'
    draft.write_text(json.dumps(document))
    identifier, existed = commit_manifest(draft, tmp_path)
    assert not existed
    committed = tmp_path / 'manifests' / 'committed' / (identifier.split(':')[1] + '.yaml')
    return tmp_path, draft, committed, document, identifier, marker


def stored_document(path):
    from talos.yaml.config import round_trip_yaml
    return round_trip_yaml().load(path.read_text())


@pytest.mark.parametrize('reference_length', [8, 64])
@pytest.mark.parametrize('section,field,value', [
    ('metadata', 'name', 'changed'),
    ('sfd', 'params', {'epochs': [2]}),
    ('uel', 'round_limit', 1),
])
def test_committed_content_changes_fail_before_caller_import(
        committed_project, monkeypatch, reference_length, section, field, value):
    root, _, committed, _, identifier, marker = committed_project
    uri = 'manifest://sha256:' + identifier.split(':')[1][:reference_length]
    assert resolve_manifest_uri(uri, root) == (committed, root)
    changed = stored_document(committed)
    changed[section][field] = value
    assert changed['lineage']['id'] == identifier
    assert canonical_manifest_id(changed) != identifier
    committed.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='content does not match'):
        resolve_manifest_uri(uri, root)
    monkeypatch.chdir(root)
    result = CliRunner().invoke(cli, ['run', '--dry-run', uri])
    assert result.exit_code == 1
    assert 'content does not match' in result.output
    assert not marker.exists()


def test_recommit_does_not_bless_changed_committed_content(committed_project):
    root, draft, committed, _, _, _ = committed_project
    index = committed.parent / 'index.json'
    original_index = index.read_bytes()
    changed = stored_document(committed)
    changed['metadata']['name'] = 'changed'
    committed.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='content does not match'):
        commit_manifest(draft, root)
    assert index.read_bytes() == original_index
    with pytest.raises(ValueError, match='content does not match'):
        fork_manifest(committed, root / 'fork.yaml', 'fork')
    assert not (root / 'fork.yaml').exists()


@pytest.mark.parametrize('source_kind', ['direct', 'symlink'])
@pytest.mark.parametrize('change', ['content', 'lineage', 'filename'])
def test_recommit_verifies_stored_source_before_new_identity(
        committed_project, source_kind, change):
    root, _, committed, _, _, _ = committed_project
    changed = stored_document(committed)
    if change == 'content':
        changed['metadata']['name'] = 'changed'
    elif change == 'lineage':
        changed['lineage']['id'] = 'sha256:' + '0' * 64
    else:
        committed = committed.with_name('0' * 64 + '.yaml')
    committed.write_text(json.dumps(changed))
    source = committed
    if source_kind == 'symlink':
        source = root / 'alias.yaml'
        source.symlink_to(committed)
    index = committed.parent / 'index.json'
    original_index = index.read_bytes()
    original_files = set(committed.parent.glob('*.yaml'))
    with pytest.raises(ValueError, match='Integrity check failed'):
        commit_manifest(source, root)
    assert index.read_bytes() == original_index
    assert set(committed.parent.glob('*.yaml')) == original_files


def test_cli_recommit_verifies_changed_stored_source_before_git(committed_project, monkeypatch):
    _, _, committed, _, _, _ = committed_project
    changed = stored_document(committed)
    changed['metadata']['name'] = 'changed'
    committed.write_text(json.dumps(changed))
    index = committed.parent / 'index.json'
    original_index = index.read_bytes()
    original_files = set(committed.parent.glob('*.yaml'))
    git_attempts = []
    def record_git_attempt(*args):
        git_attempts.append(args)
        return False
    monkeypatch.setattr('talos.cli.commands.commit.git_add_and_commit', record_git_attempt)
    result = CliRunner().invoke(cli, ['commit', str(committed)])
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert 'content does not match' in str(result.exception)
    assert not git_attempts
    assert index.read_bytes() == original_index
    assert set(committed.parent.glob('*.yaml')) == original_files


def test_unchanged_stored_source_and_alias_recommit_remain_idempotent(committed_project):
    root, _, committed, _, identifier, _ = committed_project
    original_bytes = committed.read_bytes()
    alias = root / 'alias.yaml'
    alias.symlink_to(committed)
    assert commit_manifest(committed, root) == (identifier, True)
    assert commit_manifest(alias, root) == (identifier, True)
    assert committed.read_bytes() == original_bytes


def test_reindex_reports_tampered_content_and_retains_valid_manifests(committed_project):
    root, draft, committed, document, _, _ = committed_project
    changed = stored_document(committed)
    changed['metadata']['name'] = 'changed'
    committed.write_text(json.dumps(changed))
    document['metadata']['name'] = 'valid'
    draft.write_text(json.dumps(document))
    second_id, _ = commit_manifest(draft, root)
    count, warnings = rebuild_index(root)
    assert count == 1
    assert len(warnings) == 1
    assert committed.name in warnings[0]
    assert 'content does not match' in warnings[0]
    assert warnings[0].endswith(' — skipped')
    entries = json.loads((committed.parent / 'index.json').read_text())['manifests']
    assert [entry['id'] for entry in entries] == [second_id]


@pytest.mark.parametrize('malformation', ['mixed_keys', 'circular_reference'])
def test_unhashable_committed_content_is_rejected_and_reindex_keeps_valid_entries(
        committed_project, malformation):
    from io import StringIO

    from talos.yaml.config import round_trip_yaml
    root, draft, committed, document, identifier, _ = committed_project
    document['metadata']['name'] = 'valid'
    draft.write_text(json.dumps(document))
    second_id, _ = commit_manifest(draft, root)
    index = committed.parent / 'index.json'
    original_index = index.read_bytes()
    changed = stored_document(committed)
    changed['sfd']['context'] = {1: 'integer key', 'x': 'string key'} if malformation == 'mixed_keys' else changed
    stream = StringIO()
    round_trip_yaml().dump(changed, stream)
    committed.write_text(stream.getvalue())
    for length in (8, 64):
        uri = 'manifest://sha256:' + identifier.split(':')[1][:length]
        with pytest.raises(ValueError, match='content cannot be hashed'):
            resolve_manifest_uri(uri, root)
    with pytest.raises(ValueError, match='content cannot be hashed'):
        commit_manifest(committed, root)
    with pytest.raises(ValueError, match='content cannot be hashed'):
        fork_manifest(committed, root / 'fork.yaml', 'fork')
    assert not (root / 'fork.yaml').exists()
    assert index.read_bytes() == original_index
    count, warnings = rebuild_index(root)
    assert count == 1
    assert len(warnings) == 1
    assert committed.name in warnings[0]
    assert 'content cannot be hashed' in warnings[0]
    assert warnings[0].endswith(' — skipped')
    entries = json.loads(index.read_text())['manifests']
    assert [entry['id'] for entry in entries] == [second_id]


def test_mutable_draft_changes_create_new_identity_and_keep_lineage_envelope(committed_project):
    root, draft, committed, _, identifier, _ = committed_project
    changed = stored_document(committed)
    changed['sfd']['params']['epochs'] = [1, 2]
    changed['lineage']['parent_id'] = identifier
    assert validate(changed).valid
    draft.write_text(json.dumps(changed))
    new_id, existed = commit_manifest(draft, root)
    assert not existed
    assert new_id != identifier
    path, _ = resolve_manifest_uri('manifest://' + new_id, root)
    assert stored_document(path)['lineage']['parent_id'] == identifier
    original = stored_document(committed)
    original['lineage']['committed_at'] = 'historical timestamp format'
    committed.write_text(json.dumps(original))
    assert canonical_manifest_id(original) == identifier
    assert resolve_manifest_uri('manifest://' + identifier, root)[0] == committed


@pytest.mark.parametrize('section', ['sfd', 'uel'])
def test_invalid_objective_direction_stops_compilation_and_cli(committed_project, section):
    _, draft, _, document, _, marker = committed_project
    document[section]['objective'] = {'metric': 'val_loss', 'direction': 'sideways'}
    assert [error.path for error in validate(document).errors] == [f'{section}.objective.direction']
    with pytest.raises(ValidationError, match='Expected min or max'):
        CompiledSFD(document, source_path=draft)
    draft.write_text(json.dumps(document))
    result = CliRunner().invoke(cli, ['run', '--dry-run', str(draft)])
    assert result.exit_code == 1
    assert f'{section}.objective.direction' in result.output
    assert not marker.exists()


@pytest.mark.parametrize('setting', ['save_models', 'prep_each_round', 'progress_bar'])
@pytest.mark.parametrize('value', ['false', 0])
def test_boolean_controls_reject_string_and_integer_truthiness(committed_project, setting, value):
    _, draft, _, document, _, marker = committed_project
    document['uel'][setting] = value
    assert [error.path for error in validate(document).errors] == [f'uel.{setting}']
    with pytest.raises(ValidationError, match='Must be a boolean'):
        CompiledSFD(document, source_path=draft)
    assert not marker.exists()


@pytest.mark.parametrize('objective', ['loss', {'metric': 'val_loss', 'direction': 'min'},
                                       {'direction': 'max'}, {}, None])
def test_objective_shorthand_and_inferred_metric_remain_valid(committed_project, objective):
    _, _, _, document, _, _ = committed_project
    document['sfd']['objective'] = objective
    document['uel'].update(save_models=False, prep_each_round=True, progress_bar=False)
    assert validate(document).valid


@pytest.mark.parametrize('section,field,value', [
    ('metadata', 'mode', []),
    ('uel', 'search_strategy', {'type': []}),
    ('uel', 'output_format', []),
])
def test_malformed_discriminators_return_errors_instead_of_type_errors(
        committed_project, section, field, value):
    _, _, _, document, _, _ = committed_project
    document[section][field] = value
    result = validate(document)
    assert not result.valid
    assert [error.path for error in result.errors] == [f'{section}.{field}']


def test_saved_invalid_manifest_is_rejected_before_source_hydration(committed_project, monkeypatch):
    root, draft, _, document, _, marker = committed_project
    document['sfd']['objective'] = {'metric': 'val_loss', 'direction': 'sideways'}
    run_dir = root / 'recorded'
    run_dir.mkdir()
    raw = {'yaml_reference': {'content': document, 'manifest_id': canonical_manifest_id(document),
                              'source_path': str(draft)}, 'sfd': {'module': 'caller'}}
    (run_dir / 'metadata.json').write_text(json.dumps(raw))
    def source_operation(*args, **kwargs):
        raise AssertionError('Invalid manifest reached saved-source verification or hydration')
    monkeypatch.setattr('talos.experiment.source_snapshot.verify_sources', source_operation)
    monkeypatch.setattr('talos.experiment.source_snapshot.hydrate_sources', source_operation)
    with pytest.raises(ValidationError, match='Expected min or max'):
        CompiledSFD.from_run(run_dir)
    assert not marker.exists()


@pytest.mark.parametrize('objective,path', [
    ('', 'sfd.objective'),
    (True, 'sfd.objective'),
    ({'metric': True}, 'sfd.objective.metric'),
])
def test_malformed_objective_shapes_are_rejected_before_import(committed_project, objective, path):
    _, draft, _, document, _, marker = committed_project
    document['sfd']['objective'] = objective
    assert [error.path for error in validate(document).errors] == [path]
    with pytest.raises(ValidationError):
        CompiledSFD(document, source_path=draft)
    assert not marker.exists()


@pytest.mark.parametrize('invalid_shape', [True, False])
def test_invalid_committed_envelope_fails_before_import(committed_project, invalid_shape):
    root, _, committed, _, identifier, marker = committed_project
    changed = [] if invalid_shape else stored_document(committed)
    if not invalid_shape:
        changed['lineage']['id'] = 'sha256:' + '0' * 64
    committed.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match=r'Invalid manifest format|lineage\.id does not match'):
        resolve_manifest_uri('manifest://' + identifier, root)
    assert not marker.exists()


def test_saved_content_hash_mismatch_stops_before_source_verification(committed_project, monkeypatch):
    root, draft, _, document, identifier, marker = committed_project
    document['metadata']['name'] = 'changed'
    run_dir = root / 'recorded'
    run_dir.mkdir()
    reference = {'content': document, 'manifest_id': identifier, 'source_path': str(draft)}
    (run_dir / 'metadata.json').write_text(json.dumps({'yaml_reference': reference}))
    def source_operation(*args, **kwargs):
        raise AssertionError('Changed manifest reached saved-source verification')
    monkeypatch.setattr('talos.experiment.source_snapshot.verify_sources', source_operation)
    with pytest.raises(ValueError, match='Recorded manifest content does not match its hash'):
        CompiledSFD.from_run(run_dir)
    assert not marker.exists()


@pytest.mark.parametrize('section,field,value,path', [
    ('sfd', 'module', {}, 'sfd.module'),
    ('sfd', 'params', [], 'sfd.params'),
    ('sfd', 'params', {'epochs': []}, 'sfd.params.epochs'),
    ('sfd', 'data_source', 'reader', 'sfd.data_source'),
    ('uel', 'round_limit', True, 'uel.round_limit'),
    ('uel', 'pruning_strategies', {}, 'uel.pruning_strategies'),
    ('uel', 'data_source', 'reader', 'uel.data_source'),
])
def test_existing_field_failures_remain_validation_errors(committed_project, section, field, value, path):
    _, draft, _, document, _, marker = committed_project
    document[section][field] = value
    assert [error.path for error in validate(document).errors] == [path]
    with pytest.raises(ValidationError):
        CompiledSFD(document, source_path=draft)
    assert not marker.exists()


@pytest.mark.parametrize('section', ['sfd', 'uel'])
def test_nonmapping_sections_stop_before_import(committed_project, section):
    _, draft, _, document, _, marker = committed_project
    document[section] = []
    assert [error.path for error in validate(document).errors] == [section]
    with pytest.raises(ValidationError):
        CompiledSFD(document, source_path=draft)
    assert not marker.exists()
