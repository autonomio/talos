"""One executor: real Iris callbacks, SFDs, durable records and crash recovery."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import talos
from talos.experiment import RunResult, run
from talos.parameters.ParamSpace import ParamSpace


@pytest.fixture
def iris():
    x, y = load_iris(return_X_y=True)
    a, b, c, d = train_test_split(x, y, test_size=.3, random_state=17, stratify=y)
    return {'x_train': a, 'x_val': b, 'y_train': c, 'y_val': d}


def train(x_train, y_train, x_val, y_val, params):
    model = LogisticRegression(C=params['c'], max_iter=300).fit(x_train, y_train)
    loss = log_loss(y_val, model.predict_proba(x_val))
    return SimpleNamespace(history={'loss': [loss + .1, loss], 'val_loss': [loss + .1, loss]}), model


class IrisSFD:
    @staticmethod
    def params():
        return {'c': [.1, 1.]}

    @staticmethod
    def prep(data, round_params):
        return data

    @staticmethod
    def model(data, round_params):
        return train(data['x_train'], data['y_train'], data['x_val'], data['y_val'], round_params)


def test_unchanged_scan_and_sfd_equivalence_and_durable_models(tmp_path, iris):
    scan = talos.Scan(iris['x_train'], iris['y_train'], IrisSFD.params(), train, 'iris',
                      x_val=iris['x_val'], y_val=iris['y_val'], disable_progress_bar=True,
                      experiment_dir=tmp_path / 'scan', seed=17)
    native = run(IrisSFD, iris, experiment_dir=tmp_path / 'sfd', seed=17, progress_bar=False)
    assert scan.data.c.tolist() == native.data.c.tolist()
    np.testing.assert_allclose(scan.data.val_loss, native.data.val_loss)
    assert scan.data._trial_id.tolist() == native.data._trial_id.tolist()
    np.testing.assert_array_equal(talos.Predict(scan).predict(iris['x_val'], 'val_loss', asc=True),
                                  native.predict(iris['x_val']))
    loaded = RunResult.load(scan.run_dir)
    np.testing.assert_array_equal(loaded.predict(iris['x_val'], 'val_loss', True),
                                  scan.best_model('val_loss', asc=True).predict(iris['x_val']))
    assert len(scan.round_history) == len(scan.saved_models) == len(scan.saved_weights) == 2
    assert {'start', 'end', 'duration'} <= set(scan.round_times)
    assert len(scan.learning_entropy) == 2
    scan.evaluate_models(iris['x_val'], iris['y_val'], 'multi_class', metric='val_loss', asc=True, folds=4)
    assert scan.data.eval_f1score_mean.notna().all()
    assert talos.Analyze(scan).rounds() == 2


def test_replica_trials_and_callable_numpy_parameters(tmp_path, iris):
    space = ParamSpace({'c': [np.float64(1.)], '_caller': [abs]})
    space.param_space = np.repeat(space.param_space, 3, axis=0)
    space.param_index = [0, 1, 2]
    scan = talos.Scan(iris['x_train'], iris['y_train'], space, train, 'replicas',
                      x_val=iris['x_val'], y_val=iris['y_val'], disable_progress_bar=True,
                      experiment_dir=tmp_path / 'replicas')
    assert scan.data._trial_id.nunique() == 3
    assert scan.data._param_hash.nunique() == 1
    assert scan.data._caller.tolist() == [abs] * 3
    loaded = RunResult.load(scan.run_dir)
    assert loaded.params['_caller'] == [abs]


def test_resume_exact_ids_metrics_histories_and_selection(tmp_path, iris):
    paused = run(IrisSFD, iris, experiment_dir=tmp_path / 'resume', seed=3,
                 progress_bar=False, stop_after=1)
    assert paused.status == 'paused'
    completed = run(IrisSFD, iris, experiment_dir=paused.run_dir, seed=3,
                    progress_bar=False, resume=True)
    uninterrupted = run(IrisSFD, iris, experiment_dir=tmp_path / 'complete', seed=3, progress_bar=False)
    assert completed.status == 'complete'
    assert completed.data._trial_id.tolist() == uninterrupted.data._trial_id.tolist()
    assert completed.round_history == uninterrupted.round_history
    np.testing.assert_allclose(completed.data.val_loss, uninterrupted.data.val_loss)
    np.testing.assert_array_equal(completed.predict(iris['x_val']), uninterrupted.predict(iris['x_val']))
    changed = {**iris, 'x_train': iris['x_train'] + 1}
    with pytest.raises(ValueError, match='hash'):
        run(IrisSFD, changed, experiment_dir=paused.run_dir, seed=3, progress_bar=False, resume=True)


def test_first_trial_interrupt_and_exception_are_resumable(tmp_path):
    state = {'fail': True}
    def model(data, round_params):
        if state['fail']:
            raise KeyboardInterrupt
        return {'score': round_params['n']}
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2]}, model=model)
    paused = run(sfd, experiment_dir=tmp_path / 'interrupted', progress_bar=False)
    assert paused.status == 'paused' and paused.data.empty
    assert json.loads((paused.run_dir / 'checkpoint.json').read_text())['metadata']['experiment_round'] == -1
    state['fail'] = False
    resumed = run(sfd, experiment_dir=paused.run_dir, resume=True, progress_bar=False)
    assert resumed.data.n.tolist() == [1, 2]
    def failing(data, round_params):
        raise RuntimeError('caller failure')
    sfd.model = failing
    with pytest.raises(RuntimeError, match='caller failure'):
        run(sfd, experiment_dir=tmp_path / 'failed', progress_bar=False)
    assert json.loads((tmp_path / 'failed' / 'metadata.json').read_text())['details']['status'] == 'failed'


def test_later_metrics_and_csv_quoting_are_not_lost(tmp_path):
    def model(data, round_params):
        out = {'score': round_params['n']}
        if round_params['n'] == 2:
            out['later_metric'] = .75
        return out
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2], 'label': ['a,b\nquoted']}, model=model)
    result = run(sfd, experiment_dir=tmp_path / 'variable', progress_bar=False)
    frame = pd.read_csv(result.run_dir / 'results.csv')
    assert frame.label.tolist() == ['a,b\nquoted'] * 2
    assert np.isnan(frame.later_metric.iloc[0]) and frame.later_metric.iloc[1] == .75
    assert RunResult.load(result.run_dir).data.later_metric.iloc[1] == .75


def test_native_grid_is_lazy_and_pause_control(tmp_path):
    sfd = SimpleNamespace(params=lambda: {f'p{i}': list(range(1000)) for i in range(5)},
                          model=lambda data, round_params: {'score': 1.})
    result = run(sfd, experiment_dir=tmp_path / 'large', round_limit=1, progress_bar=False)
    assert len(result.data) == 1 and result.domain.total_combinations == 10 ** 15
    def completed(event, result, params):
        if event == 'trial_completed':
            result.request_pause()
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2]}, model=lambda data, round_params: {'score': 1})
    result = run(sfd, experiment_dir=tmp_path / 'paused', event_callback=completed, progress_bar=False)
    assert result.status == 'paused' and len(result.data) == 1


def test_interventions_apply_next_trial_and_are_audited(tmp_path):
    def callback(event, result, params):
        if event == 'trial_completed' and len(result.data) == 1:
            (result.run_dir / 'interventions.json').write_text(json.dumps([{'op': 'keep_is', 'param': 'n', 'value': 3}]))
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2, 3]}, model=lambda data, round_params: {'score': 1})
    result = run(sfd, experiment_dir=tmp_path / 'controlled', event_callback=callback, progress_bar=False)
    # Files written after trial-completed are polled before the following trial.
    assert result.data.n.tolist() == [1, 3]
    assert 'keep_is' in (result.run_dir / 'audit.jsonl').read_text()


def test_result_metadata_metric_collision_fails_visibly(tmp_path):
    sfd = SimpleNamespace(params=lambda: {'n': [1]}, model=lambda data, round_params: {'start': 2})
    with pytest.raises(ValueError, match='collide'):
        run(sfd, experiment_dir=tmp_path / 'bad', progress_bar=False)


def test_core_import_has_no_dl_or_plot_imports():
    script = ('import sys, importlib.util; arrow_installed=importlib.util.find_spec("pyarrow") is not None; '
              'import talos; from talos.experiment import run; from talos.cli.main import cli; '
              'import talos.utils; '
              'assert not {"tensorflow","keras","torch","matplotlib"}.intersection(sys.modules); '
              'assert arrow_installed or "pyarrow" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_sigterm_checkpoints_pending_first_trial(tmp_path):
    sfd = tmp_path / 'slow.py'
    sfd.write_text("import os, signal\ndef params(): return {'n': [1]}\ndef prep(data, round_params): return data\ndef model(data, round_params):\n    os.kill(os.getpid(), signal.SIGTERM)\n")
    directory = tmp_path / 'signal'
    script = f'from talos import run; result=run({str(sfd)!r},experiment_dir={str(directory)!r},progress_bar=False); assert result.status == "paused"'
    executed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert json.loads((directory / 'checkpoint.json').read_text())['metadata']['experiment_round'] == -1


def test_changed_prepared_data_without_supplied_observations_is_rejected(tmp_path):
    state = {'shift': 0}
    def prep(data, round_params):
        x, y = load_iris(return_X_y=True)
        return x + state['shift'], y
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2]}, prep=prep,
                          model=lambda data, round_params: {'score': float(data[0].mean())})
    first = run(sfd, experiment_dir=tmp_path / 'prepared', stop_after=1, progress_bar=False)
    committed_checkpoint = (first.run_dir / 'checkpoint.json').read_bytes()
    committed_metadata = (first.run_dir / 'metadata.json').read_bytes()
    state['shift'] = 1
    with pytest.raises(ValueError, match='Prepared data or split changed'):
        run(sfd, experiment_dir=first.run_dir, resume=True, progress_bar=False)
    assert (first.run_dir / 'checkpoint.json').read_bytes() == committed_checkpoint
    assert (first.run_dir / 'metadata.json').read_bytes() == committed_metadata


def test_unseeded_random_resume_and_budget_configuration(tmp_path):
    from talos.experiment.param_domain import ParamDomain
    from talos.experiment.param_search import RandomStrategy
    from talos.experiment.reducer import BudgetReducer
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2, 3]}, model=lambda data, round_params: {'score': 1})
    def execute(path, **kwargs):
        return run(sfd, experiment_dir=path, search_strategy=RandomStrategy(ParamDomain(sfd.params())),
                   pruning_strategies=[BudgetReducer(max_permutations=2)], feedback_interval=1,
                   progress_bar=False, **kwargs)
    first = execute(tmp_path / 'random', stop_after=1)
    resumed = execute(first.run_dir, resume=True)
    assert len(resumed.data) == 2 and resumed.data._trial_id.nunique() == 2
    with pytest.raises(ValueError, match='hash'):
        run(sfd, experiment_dir=first.run_dir, search_strategy=RandomStrategy(ParamDomain(sfd.params())),
            pruning_strategies=[BudgetReducer(max_permutations=1)], feedback_interval=1,
            progress_bar=False, resume=True)


def test_local_strategy_controls_running_scan_and_audit(tmp_path, monkeypatch, iris):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'talos_strategy.py'
    source.write_text('def talos_strategy(scan):\n    scan.param_object.remove_is("c", 1.0)\n    return scan\n')
    scan = talos.Scan(iris['x_train'], iris['y_train'], {'c': [.1, 1., 2.]}, train, 'controlled',
                      x_val=iris['x_val'], y_val=iris['y_val'], reduction_method='local_strategy',
                      disable_progress_bar=True, experiment_dir=tmp_path / 'local')
    assert scan.data.c.tolist() == [.1, 2.]
    audit = (scan.run_dir / 'audit.jsonl').read_text()
    assert 'legacy_pending_selection' in audit and 'source_hash' in audit


def test_gamify_annotation_and_disable_in_running_scan(tmp_path, monkeypatch, iris):
    monkeypatch.chdir(tmp_path)
    def edit(event, result, params):
        if event == 'trial_completed' and len(result.data) == 1:
            control = tmp_path / 'gamify' / (result.run_dir.name + '.json')
            document = json.loads(control.read_text())
            document['0']['1'][1] = 99  # annotation must keep c=1
            document['0']['2'][0] = 'disabled'
            control.write_text(json.dumps(document))
    scan = talos.Scan(iris['x_train'], iris['y_train'], {'c': [.1, 1., 2.]}, train, 'gamify',
                      x_val=iris['x_val'], y_val=iris['y_val'], reduction_method='gamify',
                      disable_progress_bar=True, experiment_dir=tmp_path / 'game', event_callback=edit)
    assert scan.data.c.tolist() == [.1, 1.]
    assert 'legacy_pending_selection' in (scan.run_dir / 'audit.jsonl').read_text()


def test_parameter_metric_and_metadata_aliases_survive_late_schema_resume(tmp_path):
    def model(data, round_params):
        values = {'score': round_params['n']}
        if round_params['n'] == 2:
            values['loss'] = .25
        return values
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2], 'loss': ['mse'], 'start': [99]}, model=model)
    first = run(sfd, experiment_dir=tmp_path / 'aliases', stop_after=1, progress_bar=False)
    assert first.data.param__start.tolist() == [99]
    assert isinstance(first.data.start.iloc[0], str)
    resumed = run(sfd, experiment_dir=first.run_dir, resume=True, progress_bar=False)
    assert resumed.parameter_columns['loss'] == 'param__loss'
    assert resumed.data.param__loss.tolist() == ['mse', 'mse']
    assert np.isnan(resumed.data.loss.iloc[0]) and resumed.data.loss.iloc[1] == .25
    loaded = RunResult.load(resumed.run_dir)
    assert loaded.parameter_columns == resumed.parameter_columns
    np.testing.assert_array_equal(loaded.data.param__loss, resumed.data.param__loss)
    assert [record['params']['loss'] for record in loaded._records] == ['mse', 'mse']
    np.testing.assert_array_equal(pd.read_csv(resumed.run_dir / 'results.csv').param__loss, resumed.data.param__loss)


def test_native_budget_prunes_original_parameter_through_column_alias(tmp_path):
    from talos.experiment.reducer import BudgetReducer
    sfd = SimpleNamespace(params=lambda: {'loss': ['good', 'bad', 'neutral']},
                          model=lambda data, round_params: {'loss': 1. if round_params['loss'] == 'good' else 10.})
    result = run(sfd, experiment_dir=tmp_path / 'aliased-budget', progress_bar=False,
                 pruning_strategies=[BudgetReducer(max_permutations=1, metric='loss', maximize=False, trim_strategy='worst_first')],
                 feedback_interval=1)
    assert result.data.loss.tolist() == [1.]
    assert result.data.param__loss.tolist() == ['good']
    assert result.parameter_columns == {'loss': 'param__loss'}


def test_load_rejects_truncated_completed_records(tmp_path):
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2]}, model=lambda data, round_params: {'score': 1})
    result = run(sfd, experiment_dir=tmp_path / 'truncated', progress_bar=False)
    path = result.run_dir / 'round_data.jsonl'
    path.write_text(path.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='missing completed'):
        RunResult.load(result.run_dir)


def test_mixed_typed_categories_are_pruned_without_string_collision(tmp_path):
    from talos.experiment.reducer import SanityReducer
    import polars as pl
    def model(data, round_params):
        return {'score': float('nan') if type(round_params['choice']) is int else 1.0}
    sfd = SimpleNamespace(params=lambda: {'choice': [1, '1'], 'replica': [0, 1]}, model=model)
    result = run(sfd, experiment_dir=tmp_path / 'typed', progress_bar=False,
                 pruning_strategies=[SanityReducer(metric='score')], feedback_interval=1,
                 output_format='parquet')
    assert result.data.choice.tolist() == [1, '1']
    assert result.domain.values_for('choice') == ['1']
    parquet = pl.read_parquet(result.run_dir / 'results.parquet')
    assert parquet['choice'].n_unique() == 2
    assert len(parquet) == 2


def test_caller_warnings_reach_sanity_suggestions_without_changing_candidates(tmp_path):
    import warnings
    from talos.experiment.reducer import SanityReducer
    def model(data, round_params):
        if round_params['n'] == 1:
            warnings.warn('scientific warning probe', RuntimeWarning)
        return {'score': 1.0}
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2], '_warnings': ['caller']}, model=model)
    with pytest.warns(RuntimeWarning, match='scientific warning probe'):
        result = run(sfd, experiment_dir=tmp_path / 'warnings', progress_bar=False,
                     pruning_strategies=[SanityReducer(metric='score', warning_threshold=0.0)],
                     feedback_interval=1)
    assert result.data.n.tolist() == [1, 2]
    assert result.data.param___warnings.tolist() == ['caller', 'caller']
    assert result._records[0]['warnings'][0]['message'] == 'scientific warning probe'
    audit = (result.run_dir / 'audit.jsonl').read_text()
    assert 'warning rate' in audit and 'suggest' in audit


def test_python_parquet_resume_updates_summary_and_committed_parameter_aliases(tmp_path):
    import polars as pl
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2], 'loss': ['mse']},
                          model=lambda data, round_params: {'loss': 1. / round_params['n']})
    first = run(sfd, experiment_dir=tmp_path / 'summary', stop_after=1,
                output_format='parquet', progress_bar=False)
    assert len(pl.read_parquet(first.run_dir / 'results.parquet')) == 1
    resumed = run(sfd, experiment_dir=first.run_dir, resume=True,
                  output_format='parquet', progress_bar=False)
    assert len(pl.read_parquet(first.run_dir / 'results.parquet')) == 2
    metadata = json.loads((first.run_dir / 'metadata.json').read_text())
    metadata['parameter_columns'] = {}
    metadata['details']['rounds'] = 0
    (first.run_dir / 'metadata.json').write_text(json.dumps(metadata))
    loaded = RunResult.load(first.run_dir)
    assert loaded.details['rounds'] == 2
    assert loaded.parameter_columns == resumed.parameter_columns
    assert loaded.data.param__loss.tolist() == ['mse', 'mse']


def test_every_history_epoch_is_validated_before_commit_and_pending_trial_survives(tmp_path):
    state = {'bad': True}
    def model(data, round_params):
        return {'score': round_params['units'], 'history': {'loss': ['bad' if state['bad'] else 1., 1.]}}
    sfd = SimpleNamespace(params=lambda: {'units': [1, 2]}, model=model)
    directory = tmp_path / 'invalid-history'
    with pytest.raises(TypeError, match='loss.*numeric'):
        run(sfd, experiment_dir=directory, progress_bar=False)
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint['metadata']['experiment_round'] == -1
    assert checkpoint['msq_state']['yielded_count'] == 0
    assert (directory / 'round_data.jsonl').read_text() == ''
    state['bad'] = False
    resumed = run(sfd, experiment_dir=directory, resume=True, progress_bar=False)
    assert resumed.data.units.tolist() == [1, 2]
    assert resumed.data._trial_id.nunique() == 2
    assert resumed.round_history == [{'loss': [1., 1.]}] * 2


def test_failure_during_result_projection_rolls_back_staged_record_queue_and_aliases(tmp_path, monkeypatch):
    original = RunResult._refresh
    state = {'fail': True}
    def fail_once(result):
        if state['fail'] and len(result._records) == 1:
            state['fail'] = False
            raise RuntimeError('projection fault after row staging')
        return original(result)
    monkeypatch.setattr(RunResult, '_refresh', fail_once)
    sfd = SimpleNamespace(params=lambda: {'units': [1, 2], 'loss': ['mse']},
                          model=lambda data, round_params: {'loss': 1., 'score': round_params['units']})
    directory = tmp_path / 'projection'
    with pytest.raises(RuntimeError, match='projection fault'):
        run(sfd, experiment_dir=directory, progress_bar=False)
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint['metadata']['experiment_round'] == -1
    assert checkpoint['msq_state']['yielded_count'] == 0
    assert checkpoint['execution_state']['parameter_columns'] == {}
    assert (directory / 'round_data.jsonl').read_text() == ''
    resumed = run(sfd, experiment_dir=directory, resume=True, progress_bar=False)
    assert resumed.data.units.tolist() == [1, 2]
    assert resumed.data.param__loss.tolist() == ['mse', 'mse']


@pytest.mark.parametrize('metric,threshold,minimize,expected', [('score', 2., False, [1, 2]), ('loss', 1., True, [1])])
def test_native_performance_targets_stop_and_resume_without_extra_training(tmp_path, metric, threshold, minimize, expected):
    calls = []
    def model(data, round_params):
        calls.append(round_params['n'])
        return {'score': float(round_params['n']), 'loss': 1. / round_params['n']}
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2, 3]}, model=model)
    directory = tmp_path / 'target'
    result = run(sfd, experiment_dir=directory, performance_target=[metric, threshold, minimize], progress_bar=False)
    assert result.status == 'complete' and result.data.n.tolist() == expected
    resumed = run(sfd, experiment_dir=directory, performance_target=[metric, threshold, minimize], resume=True, progress_bar=False)
    assert resumed.status == 'complete' and resumed.data.n.tolist() == expected
    assert calls == expected


def test_numpy_scalar_feedback_resolves_live_candidate_and_prunes_it(tmp_path):
    observed = []
    def feedback(log, queue):
        projected = log['width'][0]
        restored = queue.resolve_log_value('width', projected)
        observed.append(type(restored))
        assert queue.remove_is('width', projected)
    sfd = SimpleNamespace(params=lambda: {'width': [np.int64(1), np.int64(2)]},
                          model=lambda data, round_params: {'score': float(round_params['width'])})
    result = run(sfd, experiment_dir=tmp_path / 'numpy-feedback', feedback_interval=1,
                 intra_callback=feedback, stop_after=1, progress_bar=False)
    assert observed == [np.int64]
    assert result.domain.values_for('width') == [np.int64(2)]
    assert result.data.width.tolist() == [1]


def test_bfloat16_torch_observations_are_passed_and_fingerprinted_without_numpy_coercion(tmp_path):
    torch = pytest.importorskip('torch')
    x, _ = load_iris(return_X_y=True)
    tensor = torch.as_tensor(x, dtype=torch.bfloat16)
    def model(data, round_params):
        assert data is tensor and data.dtype == torch.bfloat16
        return {'score': data.float().mean().item()}
    sfd = SimpleNamespace(params=lambda: {'n': [1, 2]}, model=model)
    first = run(sfd, data=tensor, experiment_dir=tmp_path / 'bfloat', stop_after=1, progress_bar=False)
    identity = first.metadata['identity']['data']
    assert identity['dtype'] == 'torch.bfloat16' and identity['shape'] == [150, 4]
    changed = tensor.clone()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match='hash'):
        run(sfd, data=changed, experiment_dir=first.run_dir, resume=True, progress_bar=False)


def test_dynamic_legacy_callback_without_module_metadata(tmp_path, iris):
    namespace = {'train': train}
    exec('def callback(x_train, y_train, x_val, y_val, params):\n'
         '    return train(x_train, y_train, x_val, y_val, params)\n', namespace)
    callback = namespace['callback']
    assert callback.__module__ is None
    scan = talos.Scan(iris['x_train'], iris['y_train'], {'c': [.1, 1.]}, callback, 'dynamic',
                      x_val=iris['x_val'], y_val=iris['y_val'], disable_progress_bar=True,
                      experiment_dir=tmp_path / 'scan')
    assert len(scan.data) == 2
    loaded = RunResult.load(scan.run_dir)
    np.testing.assert_allclose(loaded.data.val_loss, scan.data.val_loss)
    np.testing.assert_array_equal(loaded.predict(iris['x_val'], metric='val_loss', asc=True),
                                  scan.predict(iris['x_val'], metric='val_loss', asc=True))


def test_gamify_paused_edit_is_applied_before_resumed_pending_trial(tmp_path, monkeypatch, iris):
    monkeypatch.chdir(tmp_path)
    called = []
    def model(x_train, y_train, x_val, y_val, params):
        called.append(params['c'])
        return train(x_train, y_train, x_val, y_val, params)
    def execute(**options):
        return talos.Scan(iris['x_train'], iris['y_train'], {'c': [.1, 1.]}, model, 'gamify',
                          x_val=iris['x_val'], y_val=iris['y_val'], reduction_method='gamify',
                          disable_progress_bar=True, seed=17, experiment_dir=tmp_path / 'game', **options)
    paused = execute(stop_after=1)
    assert paused.status == 'paused' and called == [.1]
    first_record = (paused.run_dir / 'round_data.jsonl').read_bytes()
    control = tmp_path / 'gamify' / (paused.run_dir.name + '.json')
    document = json.loads(control.read_text())
    document['0']['1'][0] = 'disabled'
    control.write_text(json.dumps(document))
    resumed = execute(resume=True)
    assert resumed.status == 'complete' and called == [.1]
    assert resumed.data.c.tolist() == [.1]
    assert (resumed.run_dir / 'round_data.jsonl').read_bytes() == first_record
    assert 'legacy_pending_selection' in (resumed.run_dir / 'audit.jsonl').read_text()
