"""Real Iris recovery preserves realized legacy rows without another entropy request."""

import importlib.util
import json
import sys

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import talos
from talos.experiment import RunResult
from talos.experiment.artifacts import read_json, write_json
from talos.experiment.param_domain import ParamDomain
from talos.experiment.param_search.grid_strategy import GridStrategy
from talos.experiment.serialization import content_hash
from talos.parameters.ParamSpace import ParamSpace
from talos.reducers import remote_entropy
from talos.utils.exceptions import TalosDataError

_CALLER_SOURCE = """from types import SimpleNamespace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
FIT_CALLS = []
PREDICATE_CALLS = []
PREDICATE_ENABLED = True

def iris_model(x_train, y_train, x_val, y_val, params):
    FIT_CALLS.append(params['c'])
    scale = params.get('scale', 1)
    model = LogisticRegression(C=params['c'], max_iter=200).fit(x_train * scale, y_train)
    loss = log_loss(y_val, model.predict_proba(x_val * scale))
    return SimpleNamespace(history={'val_loss': [loss]}), model

def changed_model(x_train, y_train, x_val, y_val, params):
    return iris_model(x_train, y_train, x_val, y_val, {**params, 'c': params['c'] / 10})

def allow_rows(params):
    PREDICATE_CALLS.append(dict(params))
    if not PREDICATE_ENABLED:
        raise AssertionError('Boolean predicate must not run during recovery.')
    return True
"""


@pytest.fixture
def caller(tmp_path, monkeypatch):
    path = tmp_path / 'iris_caller.py'
    path.write_text(_CALLER_SOURCE)
    specification = importlib.util.spec_from_file_location('iris_resume_caller', path)
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, specification.name, module)
    specification.loader.exec_module(module)
    return module


def pause_after_one(event, result, params):
    if event == 'trial_completed' and len(result.data) == 1:
        result.request_pause()


@pytest.fixture
def scan_options(tmp_path, caller):
    x, y = load_iris(return_X_y=True)
    a, b, c, d = train_test_split(x, y, test_size=.3, random_state=17, stratify=y)
    return {'x': a, 'x_val': b, 'y': c, 'y_val': d, 'params': {'c': [.1, 1., 10.]},
            'model': caller.iris_model, 'experiment_name': 'iris-entropy', 'experiment_dir': tmp_path / 'run',
            'round_limit': 2, 'boolean_limit': caller.allow_rows, 'seed': 17,
            'disable_progress_bar': True, 'save_models': False, 'save_weights': False,
            'clear_session': False, 'reduction_metric': 'val_loss', 'minimize_loss': True}


def pause_run(options, monkeypatch, caller, method='quantum'):
    calls = []
    def initial_provider(maximum, count, selected_method):
        calls.append((maximum, count, selected_method))
        return [2, 0]
    monkeypatch.setattr(remote_entropy, 'sample_indexes', initial_provider)
    paused = talos.Scan(**options, random_method=method, event_callback=pause_after_one)
    assert paused.status == 'paused' and paused.data.c.tolist() == [10.]
    assert calls == [(3, 2, method)]
    assert caller.FIT_CALLS == [10.]
    assert len(caller.PREDICATE_CALLS) == 2
    metadata = read_json(paused.run_dir / 'metadata.json')
    state = metadata['identity']['strategy']['initial_state']
    assert state['param_space'].tolist() == [[10.], [.1]] and state['pending'] == [0, 1]
    checkpoint = read_json(paused.run_dir / 'checkpoint.json')
    assert checkpoint['msq_state']['strategy_state']['pending'] == [1]
    return paused


def no_repeat_provider(monkeypatch, mode='unavailable'):
    calls = []
    def changed_provider(maximum, count, method):
        calls.append((maximum, count, method))
        if mode == 'unavailable':
            raise TalosDataError('Offline provider is unavailable.')
        return [1, 2]
    monkeypatch.setattr(remote_entropy, 'sample_indexes', changed_provider)
    return calls


def artifact_bytes(path):
    return {str(item.relative_to(path)): item.read_bytes() for item in path.rglob('*') if item.is_file()}


@pytest.mark.parametrize('method', ['quantum', 'ambience'])
@pytest.mark.parametrize('mode', ['unavailable', 'different_entropy'])
def test_dictionary_resume_never_repeats_provider_or_boolean_predicate(scan_options, monkeypatch, caller, method, mode):
    paused = pause_run(scan_options, monkeypatch, caller, method)
    original_hash = paused.metadata['identity_hash']
    caller.PREDICATE_CALLS.clear()
    caller.PREDICATE_ENABLED = False
    calls = no_repeat_provider(monkeypatch, mode)
    resumed = talos.Scan(**scan_options, random_method=method, resume=True)
    assert calls == [] and caller.PREDICATE_CALLS == []
    assert resumed.status == 'complete' and resumed.data.c.tolist() == [10., .1]
    assert caller.FIT_CALLS == [10., .1]
    assert resumed.metadata['identity_hash'] == original_hash
    assert resumed.data._trial_id.nunique() == 2
    assert resumed.data._trial_id.iloc[0] == paused.data._trial_id.iloc[0]
    expected = [caller.iris_model(scan_options['x'], scan_options['y'], scan_options['x_val'],
                           scan_options['y_val'], {'c': value})[0].history for value in (10., .1)]
    assert resumed.round_history == expected
    np.testing.assert_allclose(resumed.data.val_loss, [history['val_loss'][0] for history in expected])
    loaded = RunResult.load(resumed.run_dir)
    assert loaded.data.c.tolist() == [10., .1] and loaded.round_history == resumed.round_history


@pytest.mark.parametrize('change', ['params', 'model', 'data', 'seed'])
def test_resume_checks_current_caller_identity_before_training(scan_options, monkeypatch, caller, change):
    paused = pause_run(scan_options, monkeypatch, caller)
    calls = no_repeat_provider(monkeypatch, 'different_entropy')
    changed = dict(scan_options)
    if change == 'params':
        changed['params'] = {'c': [.1, 1., 100.]}
    elif change == 'model':
        changed['model'] = caller.changed_model
    elif change == 'data':
        changed['x'] = scan_options['x'] + 1
    else:
        changed['seed'] = 18
    before = artifact_bytes(paused.run_dir)
    caller.PREDICATE_CALLS.clear()
    with pytest.raises(ValueError, match=r'[Hh]ash'):
        talos.Scan(**changed, random_method='quantum', resume=True)
    assert calls == [] and caller.PREDICATE_CALLS == [] and caller.FIT_CALLS == [10.]
    assert artifact_bytes(paused.run_dir) == before


@pytest.mark.parametrize('damage', ['missing_state', 'missing_rows', 'wrong_strategy', 'consumed_rows', 'non_array'])
def test_missing_or_malformed_original_state_fails_without_resampling(scan_options, monkeypatch, caller, damage):
    paused = pause_run(scan_options, monkeypatch, caller)
    path = paused.run_dir / 'metadata.json'
    metadata = json.loads(path.read_text())
    strategy = metadata['identity']['strategy']
    if damage == 'missing_state':
        del strategy['initial_state']
    elif damage == 'wrong_strategy':
        strategy['class']['qualname'] = 'GridStrategy'
    elif damage == 'missing_rows':
        del strategy['initial_state']['param_space']
    elif damage == 'consumed_rows':
        strategy['initial_state']['pending'] = [1]
    else:
        strategy['initial_state']['param_space'] = []
    path.write_text(json.dumps(metadata))
    calls = no_repeat_provider(monkeypatch)
    before = artifact_bytes(paused.run_dir)
    with pytest.raises(ValueError, match=r'[Ss]aved'):
        talos.Scan(**scan_options, random_method='quantum', resume=True)
    assert calls == [] and caller.FIT_CALLS == [10.]
    assert artifact_bytes(paused.run_dir) == before


def test_caller_created_paramspace_remains_caller_owned(scan_options, monkeypatch):
    space = ParamSpace(scan_options['params'], round_limit=2, seed=17)
    space.param_space = np.array([[10.], [.1]], dtype=object)
    space.param_index = [0, 1]
    options = {**scan_options, 'params': space}
    paused = talos.Scan(**options, event_callback=pause_after_one)
    fresh = ParamSpace(scan_options['params'], round_limit=2, seed=17)
    fresh.param_space = np.array([[10.], [.1]], dtype=object)
    fresh.param_index = [0, 1]
    calls = no_repeat_provider(monkeypatch)
    resumed = talos.Scan(**{**options, 'params': fresh}, resume=True)
    assert paused.status == 'paused' and resumed.status == 'complete'
    assert calls == [] and resumed.data.c.tolist() == [10., .1]


@pytest.mark.parametrize('scale_domain', [[1., 2.], [.1, 1., 10.]])
def test_attested_dictionary_key_order_is_checked_before_saved_rows_are_reused(scan_options, monkeypatch, caller, scale_domain):
    options = {**scan_options, 'params': {'c': [.1, 1., 10.], 'scale': scale_domain}}
    monkeypatch.setattr(remote_entropy, 'sample_indexes', lambda maximum, count, method: [0, 1])
    paused = talos.Scan(**options, random_method='quantum', event_callback=pause_after_one)
    assert paused.metadata['identity']['legacy_param_keys'] == ['c', 'scale']
    trained = list(caller.FIT_CALLS)
    caller.PREDICATE_CALLS.clear()
    calls = no_repeat_provider(monkeypatch)
    reordered = {**options, 'params': {'scale': scale_domain, 'c': [.1, 1., 10.]}}
    before = artifact_bytes(paused.run_dir)
    with pytest.raises(ValueError, match='key order'):
        talos.Scan(**reordered, random_method='quantum', resume=True)
    assert calls == [] and caller.PREDICATE_CALLS == [] and caller.FIT_CALLS == trained
    assert artifact_bytes(paused.run_dir) == before


def test_historical_identity_without_order_witness_resumes_existing_real_rows(scan_options, monkeypatch, caller):
    paused = pause_run(scan_options, monkeypatch, caller)
    # Serialize this trained checkpoint with the historical identity shape, without a new witness.
    metadata = read_json(paused.run_dir / 'metadata.json')
    del metadata['identity']['legacy_param_keys']
    historical_hash = content_hash(metadata['identity'])
    metadata['identity_hash'] = historical_hash
    metadata['details']['identity_hash'] = historical_hash
    write_json(paused.run_dir / 'metadata.json', metadata)
    checkpoint = read_json(paused.run_dir / 'checkpoint.json')
    checkpoint['metadata']['content_hash'] = historical_hash
    write_json(paused.run_dir / 'checkpoint.json', checkpoint)
    path = paused.run_dir / 'round_data.jsonl'
    record = json.loads(path.read_text())
    record['identity_hash'] = historical_hash
    path.write_text(json.dumps(record) + '\n')
    caller.PREDICATE_CALLS.clear()
    calls = no_repeat_provider(monkeypatch)
    resumed = talos.Scan(**scan_options, random_method='quantum', resume=True)
    assert resumed.status == 'complete' and resumed.data.c.tolist() == [10., .1]
    assert calls == [] and caller.PREDICATE_CALLS == [] and caller.FIT_CALLS == [10., .1]
    assert resumed.metadata['identity_hash'] == historical_hash
    assert 'legacy_param_keys' not in resumed.metadata['identity']
    assert json.loads((resumed.run_dir / 'metadata.json').read_text())['identity_hash'] == historical_hash
    assert RunResult.load(resumed.run_dir).data.c.tolist() == [10., .1]


def test_changed_order_witness_cannot_bless_a_reordered_caller(scan_options, monkeypatch, caller):
    options = {**scan_options, 'params': {'c': [.1, 1., 10.], 'scale': [1., 2.]}}
    monkeypatch.setattr(remote_entropy, 'sample_indexes', lambda maximum, count, method: [0, 1])
    paused = talos.Scan(**options, random_method='quantum', event_callback=pause_after_one)
    metadata = read_json(paused.run_dir / 'metadata.json')
    metadata['identity']['legacy_param_keys'] = ['scale', 'c']
    write_json(paused.run_dir / 'metadata.json', metadata)
    calls = no_repeat_provider(monkeypatch)
    trained = list(caller.FIT_CALLS)
    reordered = {**options, 'params': {'scale': [1., 2.], 'c': [.1, 1., 10.]}}
    before = artifact_bytes(paused.run_dir)
    with pytest.raises(ValueError, match=r'[Hh]ash'):
        talos.Scan(**reordered, random_method='quantum', resume=True)
    assert calls == [] and caller.FIT_CALLS == trained
    assert artifact_bytes(paused.run_dir) == before



def test_explicit_native_grid_strategy_retains_dictionary_scan_resume(scan_options, monkeypatch, caller):
    params = {'c': [.1, 1.]}
    options = {**scan_options, 'params': params, 'round_limit': None, 'boolean_limit': None}
    initial_strategy = GridStrategy(ParamDomain(params), seed=17)
    paused = talos.Scan(**options, search_strategy=initial_strategy, event_callback=pause_after_one)
    assert paused.status == 'paused' and paused.data.c.tolist() == [.1]
    assert paused.search_strategy is initial_strategy
    assert paused.metadata['identity']['strategy']['class']['qualname'] == 'GridStrategy'
    calls = no_repeat_provider(monkeypatch)
    fresh_strategy = GridStrategy(ParamDomain(params), seed=17)
    resumed = talos.Scan(**options, search_strategy=fresh_strategy, resume=True)
    assert resumed.status == 'complete' and resumed.data.c.tolist() == [.1, 1.]
    assert resumed.search_strategy is fresh_strategy
    assert resumed.metadata['identity_hash'] == paused.metadata['identity_hash']
    assert resumed.data._trial_id.iloc[0] == paused.data._trial_id.iloc[0]
    assert resumed.data._trial_id.nunique() == 2
    assert caller.FIT_CALLS == [.1, 1.] and calls == []
    expected = [caller.iris_model(options['x'], options['y'], options['x_val'], options['y_val'],
                                  {'c': value})[0].history for value in (.1, 1.)]
    assert resumed.round_history == expected
    assert RunResult.load(resumed.run_dir).round_history == expected
