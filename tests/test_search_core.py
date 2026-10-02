import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from talos.experiment.checkpoint_manager import CheckpointManager
from talos.experiment.errors import NonPortableValueError
from talos.experiment.msq import MSQ
from talos.experiment.param_domain import ParamDomain
from talos.experiment.param_search import GridStrategy, RandomStrategy
from talos.experiment.serialization import encode, decode, dumps, content_hash
from talos.parameters.ParamSpace import ParamSpace
from talos.parameters.DistributeParamSpace import DistributeParamSpace
from talos.reducers.GamifyMap import GamifyMap
from talos.reducers.local_strategy import local_strategy
from talos.reducers.reduce_run import reduce_run
from talos.reducers.sample_reducer import sample_reducer


def reject_large_width(params):
    return params['width'] >= 3


def test_legacy_order_ranges_and_callable_identity():
    params = {'a': [1, 2], 'z': [3, 4], 'width': (12, 48, 2), 'factory': [int]}
    facade = ParamSpace(params)
    queue = MSQ(facade.strategy, facade.strategy.domain)
    trials = list(queue)
    assert [(trial['a'], trial['z'], trial['width']) for trial in trials] == [
        (1, 3, 12), (1, 3, 30), (1, 4, 12), (1, 4, 30),
        (2, 3, 12), (2, 3, 30), (2, 4, 12), (2, 4, 30)]
    assert all(trial['factory'] is int for trial in trials)


def test_replicates_get_distinct_trial_ids():
    facade = ParamSpace({'width': [1, 1]})
    trials = list(MSQ(facade.strategy, facade.strategy.domain))
    assert len(trials) == 2
    assert trials[0]['_id'] == trials[1]['_id']
    assert trials[0]['_trial_id'] != trials[1]['_trial_id']


def test_joint_constraints_pending_mutation_and_exhaustion():
    facade = ParamSpace({'width': [1, 2, 3], 'depth': [1, 2]}, boolean_limit=lambda p: p['width'] * p['depth'] <= 3)
    queue = MSQ(facade.strategy, facade.strategy.domain)
    assert next(queue)['width'] == 1
    facade.remove_is('width', 3)
    facade.remove_lambda(lambda p: p['width'] >= 1)
    facade.param_index.reverse()
    assert [(trial['width'], trial['depth']) for trial in queue] == [(2, 1), (1, 2)]
    domain = ParamDomain({'width': [1]})
    queue = MSQ(GridStrategy(domain), domain)
    queue.remove_is('width', 1)
    assert list(queue) == []


def test_finite_random_exhaustion():
    domain = ParamDomain({'width': [1, 2, 3]})
    queue = MSQ(RandomStrategy(domain, seed=17), domain, n_permutations=8)
    assert sorted(trial['width'] for trial in queue) == [1, 2, 3]


def test_seeded_sharding_is_disjoint_and_complete():
    params = {'width': [1, 2, 3], 'depth': [1, 2]}
    first = DistributeParamSpace(params, machines=4, seed=17)
    second = DistributeParamSpace(params, machines=4, seed=17)
    seen = []
    for worker in first.param_spaces:
        assert first.param_spaces[worker].param_space.tolist() == second.param_spaces[worker].param_space.tolist()
        seen.extend(tuple(row) for row in first.param_spaces[worker].param_space.tolist())
    assert len(seen) == len(set(seen)) == 6
    assert set(seen) == {(width, depth) for width in [1, 2, 3] for depth in [1, 2]}


@pytest.mark.parametrize('method', ['uniform_mersenne', 'uniform_crypto', 'sobol', 'halton', 'korobov_matrix', 'latin_matrix', 'latin_improved', 'latin_sudoku'])
def test_sampler_candidates_are_legal_and_unique(method):
    result = sample_reducer(7, 31, method, seed=17)
    assert len(result) == len(set(result)) == 7
    assert all(0 <= index < 31 for index in result)
    if method != 'uniform_crypto':
        assert result == sample_reducer(7, 31, method, seed=17)


def test_serialization_preserves_typed_values_and_flags_local_functions():
    values = {'optimizer': int, 'tuple': (1, 2), 'array': np.array([1, 2], dtype=np.int32), 'nan': float('nan'), 'scalar': np.int64(5)}
    restored = decode(json.loads(dumps(values)))
    assert restored['optimizer'] is int
    assert restored['tuple'] == (1, 2)
    assert restored['array'].dtype == np.int32
    assert np.array_equal(restored['array'], values['array'])
    assert isinstance(restored['scalar'], np.int64)
    assert np.isnan(restored['nan'])
    assert content_hash(values) == content_hash(dict(reversed(list(values.items()))))
    marker = encode(lambda params: params)
    assert marker['portable'] is False
    with pytest.raises(NonPortableValueError, match='nonportable'):
        decode(marker)


def test_checkpoint_roundtrip_keeps_pending_replicates_and_predicates(tmp_path):
    params = {'width': [1, 1, 2, 3], 'factory': [int]}
    facade = ParamSpace(params)
    queue = MSQ(facade.strategy, facade.strategy.domain)
    queue.remove_custom(reject_large_width)
    next(queue)
    manager = CheckpointManager(checkpoint_interval=1)
    manager.save(tmp_path, queue, facade.strategy.domain, 0, 4, strategy_type='LegacyStrategy', content_hash=content_hash(params))
    fresh = ParamSpace(params)
    resumed = MSQ(fresh.strategy, fresh.strategy.domain)
    checkpoint = manager.validate(tmp_path, strategy_type='LegacyStrategy', content_hash=content_hash(params))
    fresh.strategy.domain.set_state(checkpoint['domain_state'])
    resumed.set_state(checkpoint['msq_state'])
    assert [(trial['width'], trial['factory']) for trial in resumed] == [(1, int), (2, int)]


def test_checkpoint_nonportable_predicate_refuses_resume(tmp_path):
    domain = ParamDomain({'width': [1, 2]})
    queue = MSQ(GridStrategy(domain), domain)
    queue.remove_custom(lambda p: p['width'] > 1)
    manager = CheckpointManager()
    manager.save(tmp_path, queue, domain, 0, 2, strategy_type='GridStrategy', content_hash=content_hash(domain.params))
    with pytest.raises(NonPortableValueError, match='nonportable'):
        manager.load(tmp_path)


def test_hot_edit_local_strategy_applies_next_round(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'talos_strategy.py'
    path.write_text('def talos_strategy(scan):\n    scan.value = 1\n    return scan\n')
    scan = SimpleNamespace()
    local_strategy(scan)
    assert scan.value == 1
    path.write_text('def talos_strategy(scan):\n    scan.value = 2\n    return scan\n')
    local_strategy(scan)
    assert scan.value == 2


def test_gamify_annotations_do_not_prune(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    facade = ParamSpace({'width': (12, 48, 2)})
    scan = SimpleNamespace(param_object=facade, experiment_name='experiment', _experiment_id='run')
    control = GamifyMap(scan)
    control.updated_dict = json.loads(json.dumps(control.gamify_dict))
    control.updated_dict['0']['0'][1] = .75
    control.run_updates()
    assert len(facade.param_index) == 2
    control.updated_dict['0']['1'][0] = 'disabled'
    control.run_updates()
    assert [facade.param_space[index, 0] for index in facade.param_index] == [12]


def test_reducer_cadence_means_completed_rounds():
    calls = []
    facade = ParamSpace({'width': [1, 2, 3, 4, 5, 6]})
    scan = SimpleNamespace(param_object=facade, performance_target=None, reduction_method=lambda context: calls.append(context.param_object.round_counter), reduction_interval=3, pbar=SimpleNamespace(update=lambda count: None))
    for count in range(1, 7):
        facade.round_counter = count
        reduce_run(scan)
    assert calls == [3, 6]


def test_categorical_correlation_honors_loss_direction():
    from sklearn.datasets import load_iris
    from talos.reducers.correlation import correlation
    iris = load_iris()
    frame = pd.DataFrame({'species': iris.target, 'sepal_length': iris.data[:, 0]})
    def context(minimize):
        return SimpleNamespace(result=frame, reduction_window=len(frame), reduction_metric='sepal_length',
            _param_dict_keys=['species'], minimize_loss=minimize, reduction_threshold=.2,
            param_object=ParamSpace({'species': [0, 1, 2]}))
    maximize = context(False)
    correlation(maximize, 'spearman')
    assert 0 not in [maximize.param_object.param_space[index, 0] for index in maximize.param_object.param_index]
    minimize = context(True)
    correlation(minimize, 'spearman')
    assert 2 not in [minimize.param_object.param_space[index, 0] for index in minimize.param_object.param_index]


def test_legacy_strategy_accepts_new_domain_values_without_replaying_trials():
    facade = ParamSpace({'width': [1, 2], 'depth': [1]})
    queue = MSQ(facade.strategy, facade.strategy.domain)
    assert next(queue)['width'] == 1
    queue.inject_value('width', 3)
    assert [trial['width'] for trial in queue] == [2, 3]


def test_generic_checkpoint_dedup_survives_without_external_log(tmp_path):
    domain = ParamDomain({'width': [1, 2, 3]})
    queue = MSQ(GridStrategy(domain), domain)
    assert next(queue)['width'] == 1
    manager = CheckpointManager()
    manager.save(tmp_path, queue, domain, 0, 3, strategy_type='GridStrategy', content_hash=content_hash(domain.params))
    state = manager.load(tmp_path)
    fresh_domain = ParamDomain({'width': [1, 2, 3]})
    fresh = MSQ(GridStrategy(fresh_domain), fresh_domain)
    fresh_domain.set_state(state['domain_state'])
    fresh.set_state(state['msq_state'])
    fresh_domain.inject_value('width', 4)
    assert [trial['width'] for trial in fresh] == [2, 3, 4]


def test_random_joint_filter_can_validly_exhaust_domain():
    domain = ParamDomain({'width': [1, 2], 'depth': [1, 2]})
    queue = MSQ(RandomStrategy(domain, seed=17), domain)
    queue.remove_custom(lambda p: p['width'] * p['depth'] > 0)
    assert list(queue) == []


def test_numpy_array_parameter_remains_an_object():
    array = np.array([1, 2], dtype=np.float64)
    facade = ParamSpace({'array': [array]})
    queue = MSQ(facade.strategy, facade.strategy.domain)
    assert next(queue)['array'] is array
    restored = decode(json.loads(dumps(facade.param_space)))
    assert restored.shape == (1, 1)
    assert np.array_equal(restored[0, 0], array)


def test_legacy_control_changes_enter_core_intervention_log():
    facade = ParamSpace({'width': [1, 2]})
    queue = MSQ(facade.strategy, facade.strategy.domain)
    facade.remove_is('width', 2)
    assert queue.intervention_log[-1]['operation'] == 'remove_is'
    assert queue.intervention_log[-1]['source'] == 'legacy_reducer'
    assert queue.intervention_log[-1]['value'] == 2


def test_native_domain_normalization_does_not_materialize_grid():
    from talos.parameters.ParamSpace import normalize_domains
    domains = normalize_domains({f'parameter_{index}': [1, 2, 3] for index in range(100)})
    assert len(domains) == 100
    assert all(values == [1, 2, 3] for values in domains.values())


def test_inherited_limen_reducers_keep_analysis_and_state_contract():
    import polars as pl
    from sklearn.datasets import load_iris
    from talos.experiment.reducer import BudgetReducer, CorrelationReducer, FocusReducer, SanityReducer, SaturationReducer
    from talos.experiment.reducer.registry import REDUCER_REGISTRY
    assert set(REDUCER_REGISTRY) == {'budget', 'correlation', 'focus', 'sanity', 'saturation'}
    iris = load_iris()
    log = pl.DataFrame({'species': iris.target, 'sepal_length': iris.data[:, 0]})
    domain = ParamDomain({'species': [0, 1, 2]})
    queue = MSQ(GridStrategy(domain), domain)
    reducers = [BudgetReducer(max_permutations=4),
                CorrelationReducer(metric='sepal_length', min_observations=10, n_boot=10),
                FocusReducer(metric='sepal_length', breakthrough_threshold=6.5),
                SanityReducer(metric='sepal_length'),
                SaturationReducer(metric='sepal_length', min_samples_per_value=5, window_size=10)]
    for reducer in reducers:
        interventions = reducer.analyze_and_intervene(log, queue)
        assert isinstance(interventions, list)
        assert all('op' in intervention for intervention in interventions)
        state = reducer.get_state()
        reducer.set_state(state)
        assert reducer.get_state() == state


@pytest.mark.parametrize('method', ['trees', 'forrest'])
def test_legacy_tree_reducers_preserve_categorical_objective_direction(method):
    from sklearn.datasets import load_iris
    from talos.reducers.trees import trees
    from talos.reducers.forrest import forrest
    iris = load_iris()
    frame = pd.DataFrame({'species': iris.target, 'sepal_length': iris.data[:, 0]})
    for minimize in [False, True]:
        facade = ParamSpace({'species': [0, 1, 2]})
        context = SimpleNamespace(result=frame, reduction_window=len(frame), reduction_metric='sepal_length',
            _param_dict_keys=['species'], minimize_loss=minimize, seed=17, param_object=facade)
        (trees if method == 'trees' else forrest)(context)
        retained = [facade.param_space[index, 0] for index in facade.param_index]
        removed = set([0, 1, 2]) - set(retained)
        assert len(removed) == 1
        mean = frame[frame['species'].isin(removed)]['sepal_length'].mean()
        assert mean > frame['sepal_length'].mean() if minimize else mean < frame['sepal_length'].mean()


def test_runtime_callable_capture_stays_live_and_resume_is_explicitly_nonportable():
    state = object()
    def local_activation(value):
        return value if state is not None else None
    facade = ParamSpace({'activation': [local_activation]})
    trial = next(MSQ(facade.strategy, facade.strategy.domain))
    assert trial['activation'] is local_activation
    assert trial['activation'](3) == 3
    marker = encode(local_activation)
    assert marker['portable'] is False
    assert marker['closure'][0]['__talos_type__'] == 'nonportable_state'
    with pytest.raises(NonPortableValueError, match='nonportable'):
        decode(marker)


def test_recursive_local_callable_hash_does_not_recurse():
    def recursive(value):
        return recursive(value - 1) if value else 0
    assert len(content_hash(recursive)) == 64
    assert recursive(3) == 0


def test_finite_filtered_legacy_queue_never_fills_unselected_rows():
    facade = ParamSpace({'width': [1, 2, 3], 'depth': [1, 2]})
    facade.param_index = [0, 1]
    queue = MSQ(facade.strategy, facade.strategy.domain, max_filter_retries=1)
    queue.remove_custom(lambda params: params['width'] == 1)
    assert list(queue) == []


def test_filtered_priority_queue_is_iterative_and_empty_distribution_valid():
    domain = ParamDomain({'width': [1]})
    queue = MSQ(GridStrategy(domain), domain)
    queue.remove_custom(lambda params: True)
    for _ in range(1500):
        queue.inject({'width': 1})
    assert list(queue) == []
    queue.remove_is('width', 1)
    assert queue.distribution() == {'width': {}}
    assert queue.distribution('width') == {}


def test_legacy_injected_domain_respects_joint_boolean_constraint():
    facade = ParamSpace({'width': [1], 'depth': [1, 2]}, boolean_limit=lambda params: params['width'] * params['depth'] <= 2)
    queue = MSQ(facade.strategy, facade.strategy.domain)
    next(queue)
    queue.inject_value('width', 2)
    assert facade.strategy.domain.values_for('width') == [1, 2]
    assert [(trial['width'], trial['depth']) for trial in queue] == [(1, 2), (2, 1)]


def test_named_filters_support_typed_numpy_candidates_after_checkpoint(tmp_path):
    from talos.experiment.reducer.filter_types import FILTER_BUILDERS, FILTER_KEEP_VALUES
    original = np.array([1, 2])
    other = np.array([3, 4])
    domain = ParamDomain({'weights': [original, other]})
    queue = MSQ(GridStrategy(domain), domain)
    settings = {'param': 'weights', 'values': [original]}
    queue.set_filter('weights', FILTER_BUILDERS[FILTER_KEEP_VALUES](settings),
                     filter_type=FILTER_KEEP_VALUES, filter_params=settings)
    manager = CheckpointManager()
    manager.save(tmp_path, queue, domain, -1, 2, strategy_type='GridStrategy', content_hash='test')
    state = manager.load(tmp_path)
    restored_domain = ParamDomain({'weights': [original, other]})
    restored_domain.set_state(state['domain_state'])
    restored = MSQ(GridStrategy(restored_domain), restored_domain)
    restored.set_state(state['msq_state'])
    assert len(list(restored)) == 1


def test_native_feedback_resolves_callable_and_array_categories_from_real_iris_log():
    import polars as pl
    from sklearn.datasets import load_iris
    from talos.experiment.feedback_controller import FeedbackController
    from talos.experiment.reducer.focus_reducer import FocusReducer
    iris = load_iris()
    low = iris.data[iris.target == 0].mean(axis=0)
    high = iris.data[iris.target == 2].mean(axis=0)
    domain = ParamDomain({'activation': [int, float], 'weights': [low, high]})
    queue = MSQ(GridStrategy(domain), domain)
    log = pl.DataFrame({'activation': [dumps(int if species == 2 else float) for species in iris.target],
                        'weights': [dumps(high if species == 2 else low) for species in iris.target],
                        'score': iris.data[:, 0]})
    focus = FocusReducer(metric='score', breakthrough_threshold=7, min_observations=3)
    feedback = FeedbackController(pruning_strategies=[focus])
    applied = feedback.trigger(log, queue, queue._strategy, 150)
    assert len(applied) == 2
    trials = list(queue)
    assert len(trials) == 1
    assert trials[0]['activation'] is int
    assert trials[0]['weights'] is high


def test_feedback_resolves_scalar_combo_values_and_preserves_literal_strings():
    from talos.experiment.feedback_controller import FeedbackController
    weights = np.array([1, 2])
    literal = dumps(int)
    domain = ParamDomain({'activation': [int, float], 'weights': [weights], 'label': [literal]})
    queue = MSQ(GridStrategy(domain), domain)
    assert queue.resolve_log_value('label', literal) is literal
    FeedbackController._apply_intervention(queue, {'op': 'remove_is', 'param': 'activation', 'value': dumps(float)})
    FeedbackController._apply_intervention(queue, {'op': 'inject', 'combo':
        {'activation': dumps(int), 'weights': dumps(weights), 'label': literal}, 'prioritize': True})
    trial = next(queue)
    assert trial['activation'] is int
    assert trial['weights'] is weights
    assert trial['label'] == literal
    assert domain.values_for('activation') == [int]


def test_scientific_numpy_scalar_and_structured_array_roundtrip():
    from datetime import datetime, timezone, timedelta
    from sklearn.datasets import load_iris
    iris = load_iris()
    structured = np.array(list(zip(iris.data[:, 0], iris.target)), dtype=[('sepal_length', 'f8'), ('species', 'i4')])
    values = [np.datetime64('2026-10-01T12:34:56.123456'), np.timedelta64(1234, 'us'),
              np.longdouble(iris.data[0, 0]), np.complex128(iris.data[0, 0] + 1j * iris.data[0, 1]),
              np.bytes_('iris'), structured, structured[0],
              datetime(2026, 10, 1, tzinfo=timezone.utc), timedelta(days=1, seconds=4)]
    restored = decode(json.loads(dumps(values)))
    for original, recovered in zip(values, restored):
        if isinstance(original, np.ndarray):
            assert original.dtype == recovered.dtype
            assert np.array_equal(original, recovered)
        else:
            assert type(original) is type(recovered)
            assert original == recovered


def test_numpy_exported_callable_is_portable():
    marker = encode(np.tanh)
    assert marker['portable'] is True
    assert decode(marker) is np.tanh


@pytest.mark.parametrize('method', ['spearman', 'trees', 'forrest'])
def test_legacy_reducers_keep_parameter_loss_distinct_from_objective_loss(method):
    from sklearn.datasets import load_iris
    from talos.reducers.correlation import correlation
    from talos.reducers.trees import trees
    from talos.reducers.forrest import forrest
    iris = load_iris()
    candidates = ['setosa_loss', 'versicolor_loss', 'virginica_loss']
    frame = pd.DataFrame({'param__loss': [candidates[index] for index in iris.target], 'loss': iris.data[:, 0]})
    for minimize in [False, True]:
        facade = ParamSpace({'loss': candidates})
        context = SimpleNamespace(result=frame, reduction_window=len(frame), reduction_metric='loss',
            _param_dict_keys=['loss'], parameter_columns={'loss': 'param__loss'}, minimize_loss=minimize,
            seed=17, reduction_threshold=.2, param_object=facade)
        if method == 'spearman':
            correlation(context)
        else:
            (trees if method == 'trees' else forrest)(context)
        retained = [facade.param_space[index, 0] for index in facade.param_index]
        removed = set(candidates) - set(retained)
        assert len(removed) == 1
        objective_mean = frame[frame['param__loss'].isin(removed)]['loss'].mean()
        assert objective_mean > frame['loss'].mean() if minimize else objective_mean < frame['loss'].mean()


def test_numpy_category_distribution_remains_available_for_budget_analysis():
    values = [np.array([1, 2]), np.array([3, 4])]
    domain = ParamDomain({'weights': values})
    queue = MSQ(GridStrategy(domain), domain)
    assert queue.distribution('weights') == {dumps(value): 1 for value in values}
    assert queue.distribution() == {'weights': {dumps(value): 1 for value in values}}


def _local_audit_iris_model(x_train, y_train, x_val, y_val, params):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    fitted = LogisticRegression(C=params['c'], max_iter=300).fit(x_train, y_train)
    loss = log_loss(y_val, fitted.predict_proba(x_val))
    return SimpleNamespace(history={'val_loss': [loss]}), fitted


def _local_audit_alternate_iris_model(x_train, y_train, x_val, y_val, params):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    fitted = LogisticRegression(C=params['c'] * .01, max_iter=300).fit(x_train, y_train)
    loss = log_loss(y_val, fitted.predict_proba(x_val))
    return SimpleNamespace(history={'val_loss': [loss]}), fitted


@pytest.fixture
def local_audit_models(tmp_path, monkeypatch):
    """Snapshot real caller models without the unrelated acceptance-test import graph."""
    import importlib.util
    import inspect
    import sys
    path = tmp_path / 'talos_audit_models.py'
    source = 'from types import SimpleNamespace\n\n' + '\n'.join(
        inspect.getsource(model) for model in (_local_audit_iris_model, _local_audit_alternate_iris_model))
    path.write_text(source)
    specification = importlib.util.spec_from_file_location('talos_audit_models', path)
    assert specification is not None and specification.loader is not None
    models = importlib.util.module_from_spec(specification)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, specification.name, models)
    specification.loader.exec_module(models)
    return models


def test_hot_edit_live_controls_are_audited_without_pending_queue_changes(tmp_path, monkeypatch, local_audit_models):
    import hashlib
    import talos
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    monkeypatch.chdir(tmp_path)
    x, y = load_iris(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.3, random_state=17, stratify=y)
    path = tmp_path / 'talos_strategy.py'
    first = 'def talos_strategy(scan):\n    scan.reduction_threshold = .4\n    scan.annotation = 1\n    return scan\n'
    second = ('from talos_audit_models import _local_audit_alternate_iris_model\n'
              'def talos_strategy(scan):\n    scan.reduction_threshold = .8\n'
              '    scan.model = _local_audit_alternate_iris_model\n'
              '    scan.annotation = getattr(scan, "annotation", 0) + 1\n    return scan\n')
    path.write_text(first)
    def edit(event, result, params):
        if event == 'trial_completed' and len(result.data) == 1:
            path.write_text(second)
    scan = talos.Scan(x_train, y_train, {'c': [.1, 1., 10.]}, local_audit_models._local_audit_iris_model, 'controlled',
        x_val=x_val, y_val=y_val, reduction_method='local_strategy', disable_progress_bar=True,
        experiment_dir=tmp_path / 'run', save_models=False, save_weights=False,
        clear_session=False, event_callback=edit, reduction_metric='val_loss', minimize_loss=True)
    assert scan.data.c.tolist() == [.1, 1., 10.]
    expected, _ = local_audit_models._local_audit_alternate_iris_model(x_train, y_train, x_val, y_val, {'c': 10.})
    assert scan.data.val_loss.iloc[-1] == expected.history['val_loss'][-1]
    entries = [json.loads(line) for line in (scan.run_dir / 'audit.jsonl').read_text().splitlines()]
    interventions = [item for entry in entries for item in entry['interventions']]
    controls = [item for item in interventions if item.get('op', item.get('operation')) == 'legacy_local_control_change']
    revisions = [item for item in interventions if item.get('op', item.get('operation')) == 'legacy_local_source_revision']
    assert len(controls) == len(revisions) == 2
    assert controls[0]['changes']['reduction_threshold'] == {'before': .2, 'after': .4}
    assert controls[1]['changes']['reduction_threshold'] == {'before': .4, 'after': .8}
    assert 'model' in controls[1]['changes']
    assert all('annotation' not in item['changes'] for item in controls)
    assert not any(item.get('op', item.get('operation')) == 'legacy_pending_selection' for item in interventions)
    for revision, source in zip(revisions, [first, second]):
        assert revision['source_hash'] == hashlib.sha256(source.encode()).hexdigest()
        assert (scan.run_dir / revision['snapshot_path']).read_text() == source


def test_distributed_replicates_have_unique_stable_checkpointed_trial_ids(tmp_path):
    params = {'width': [1, 1, 1, 1]}
    first = DistributeParamSpace(params, machines=2, seed=17)
    regenerated = DistributeParamSpace(params, machines=2, seed=17)
    seen = []
    for worker, facade in first.param_spaces.items():
        queue = MSQ(facade.strategy, facade.strategy.domain)
        initial = next(queue)
        directory = tmp_path / str(worker)
        directory.mkdir()
        manager = CheckpointManager()
        state = {'parameter_columns': {'width': 'width'}, 'details': {'worker': worker}}
        manager.save(directory, queue, facade.strategy.domain, 0, 2,
            strategy_type='LegacyStrategy', content_hash=content_hash(params), execution_state=state)
        checkpoint = manager.validate(directory, strategy_type='LegacyStrategy', content_hash=content_hash(params))
        assert checkpoint['execution_state'] == state
        fresh = ParamSpace(params)
        resumed = MSQ(fresh.strategy, fresh.strategy.domain)
        fresh.strategy.domain.set_state(checkpoint['domain_state'])
        resumed.set_state(checkpoint['msq_state'])
        assert fresh.shard_id == worker and fresh.shard_namespace == facade.shard_namespace
        trials = [initial, *list(resumed)]
        replica = regenerated.param_spaces[worker]
        stable = list(MSQ(replica.strategy, replica.strategy.domain))
        assert [trial['_trial_id'] for trial in trials] == [trial['_trial_id'] for trial in stable]
        seen.extend(trials)
    assert len({trial['_trial_id'] for trial in seen}) == 4
    assert len({trial['_id'] for trial in seen}) == 1
    unsharded = ParamSpace(params)
    for occurrence, trial in enumerate(MSQ(unsharded.strategy, unsharded.strategy.domain)):
        assert trial['_trial_id'] == content_hash({'parameters': trial['_id'], 'occurrence': occurrence})


def test_gamify_absolute_experiment_path_writes_legacy_sibling_file(tmp_path):
    facade = ParamSpace({'width': [12, 24]})
    scan = SimpleNamespace(param_object=facade, experiment_name=str(tmp_path / 'experiment'), _experiment_id='run')
    control = GamifyMap(scan)
    control.export_json()
    expected = tmp_path / 'experiment' / 'run.json'
    assert Path(control._filename + '.json') == expected
    assert json.loads(expected.read_text()) == control.gamify_dict


@pytest.mark.parametrize('inline_model', [False, True])
def test_hot_edit_controls_model_and_source_survive_scan_pause_resume(tmp_path, monkeypatch, inline_model, local_audit_models):
    import talos
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    monkeypatch.chdir(tmp_path)
    x, y = load_iris(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.3, random_state=17, stratify=y)
    path = tmp_path / 'talos_strategy.py'
    first = 'def talos_strategy(scan):\n    scan.reduction_threshold = .4\n    return scan\n'
    second = ('from talos_audit_models import _local_audit_alternate_iris_model\n'
              'def talos_strategy(scan):\n    scan.reduction_threshold = .8\n'
              '    scan.model = _local_audit_alternate_iris_model\n    return scan\n')
    if inline_model:
        second = second.replace('def talos_strategy(scan):',
            'def replacement_model(*args):\n    return _local_audit_alternate_iris_model(*args)\n'
            'def talos_strategy(scan):').replace('scan.model = _local_audit_alternate_iris_model',
                                                 'scan.model = replacement_model')
    def edit(event, result, params):
        if event == 'trial_completed' and len(result.data) == 1:
            path.write_text(second)
    def execute(directory, **options):
        return talos.Scan(x_train, y_train, {'c': [.1, 1., 10.]}, local_audit_models._local_audit_iris_model, 'controlled',
            x_val=x_val, y_val=y_val, reduction_method='local_strategy', disable_progress_bar=True,
            experiment_dir=directory, save_models=False, save_weights=False, seed=17,
            clear_session=False, event_callback=edit, reduction_metric='val_loss', minimize_loss=True, **options)
    path.write_text(first)
    paused = execute(tmp_path / 'resume', stop_after=2)
    assert paused.status == 'paused'
    live = CheckpointManager().load(paused.run_dir)['execution_state']['legacy_live_controls']
    assert live['controls']['reduction_threshold'] == .8
    assert live['model_reference']['portable'] is True
    assert live['model_sources']['modules']
    import sys
    for name in list(live['model_sources']['modules']):
        if name.startswith('_talos_local_strategy_'):
            monkeypatch.delitem(sys.modules, name, raising=False)
    resumed = execute(paused.run_dir, resume=True)
    path.write_text(first)
    uninterrupted = execute(tmp_path / 'complete')
    assert resumed.status == uninterrupted.status == 'complete'
    assert resumed.reduction_threshold == uninterrupted.reduction_threshold == .8
    if inline_model:
        assert resumed.model.__name__ == uninterrupted.model.__name__ == 'replacement_model'
    else:
        assert resumed.model is uninterrupted.model is local_audit_models._local_audit_alternate_iris_model
    assert resumed.data._trial_id.tolist() == uninterrupted.data._trial_id.tolist()
    assert resumed.round_history == uninterrupted.round_history
    np.testing.assert_array_equal(resumed.data.val_loss, uninterrupted.data.val_loss)


class FocusNullableIrisSFD:
    @staticmethod
    def params():
        return {'c': [.1, None, 1.]}

    @staticmethod
    def prep(data, params):
        return data

    @staticmethod
    def model(data, params):
        effective = {'c': .5 if params['c'] is None else params['c']}
        return _local_audit_iris_model(data['x_train'], data['y_train'], data['x_val'], data['y_val'], effective)


class FocusNullableIntegralIrisSFD(FocusNullableIrisSFD):
    @staticmethod
    def params():
        return {'c': [np.int64(3), None, np.int64(10)]}


@pytest.mark.parametrize('sfd,expected', [(FocusNullableIrisSFD, [.08, .12]),
                                        (FocusNullableIntegralIrisSFD, [2, 4])])
def test_focus_resolves_nullable_numeric_categories_before_interpolation(tmp_path, sfd, expected):
    from talos.experiment import run
    from talos.experiment.reducer.focus_reducer import FocusReducer
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    x, y = load_iris(return_X_y=True)
    a, b, c, d = train_test_split(x, y, test_size=.3, random_state=17, stratify=y)
    data = {'x_train': a, 'x_val': b, 'y_train': c, 'y_val': d}
    reducer = FocusReducer(metric='val_loss', breakthrough_threshold=1., maximize=False,
        min_observations=1, variation_count=3)
    result = run(sfd, data, experiment_dir=tmp_path / 'focus', progress_bar=False,
        seed=17, feedback_interval=1, pruning_strategies=[reducer], n_permutations=4)
    assert len(result.data) >= 2
    assert None not in result.data.c.tolist()
    values = [value for value in result.domain.values_for('c') if value is not None]
    assert all(any(np.isclose(value, expected_value) for value in values) for expected_value in expected)
    assert reducer._breakthrough_combo['c'] == result.data.loc[result.data.val_loss.idxmin(), 'c']


@pytest.mark.parametrize('method', ['quantum', 'ambience'])
def test_external_entropy_refills_duplicate_invalid_and_inclusive_indexes(method, monkeypatch):
    from talos.reducers import remote_entropy
    calls = []
    def provider(maximum, count, selected_method):
        assert selected_method == method
        calls.append((maximum, count))
        return [np.int64(0), 0, 31, -1, '2', True, 1] if len(calls) == 1 else [2]
    monkeypatch.setattr(remote_entropy, 'sample_indexes', provider)
    assert sample_reducer(3, 31, method) == [0, 1, 2]
    assert calls == [(31, 3), (31, 1)]


@pytest.mark.parametrize('method', ['quantum', 'ambience'])
def test_external_entropy_fails_explicitly_if_unique_population_is_unavailable(method, monkeypatch):
    from talos.reducers import remote_entropy
    from talos.utils.exceptions import TalosDataError
    calls = []
    def provider(maximum, count, selected_method):
        assert maximum == 31 and selected_method == method
        calls.append(count)
        return [0, 0]
    monkeypatch.setattr(remote_entropy, 'sample_indexes', provider)
    with pytest.raises(TalosDataError, match='unique indexes'):
        sample_reducer(2, 31, method)
    assert len(calls) == 32


@pytest.mark.parametrize('initial_flags,change', [
    ({'save_models': False, 'save_weights': True}, 'scan.save_weights = False'),
    ({'save_models': False, 'save_weights': False}, 'scan.save_models = True')])
def test_live_array_reassignment_and_save_flags_resume_exactly(tmp_path, monkeypatch, capsys, initial_flags, change, local_audit_models):
    import talos
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from talos.experiment.provenance import fingerprint
    monkeypatch.chdir(tmp_path)
    x, y = load_iris(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.3, random_state=17, stratify=y)
    source = ('def talos_strategy(scan):\n'
              '    if scan.param_object.round_counter == 1:\n'
              '        scan.x_train = scan.x_train + 1\n'
              f'        {change}\n'
              '        scan.print_params = True\n'
              '        scan.clear_session = False\n'
              '    return scan\n')
    (tmp_path / 'talos_strategy.py').write_text(source)
    def execute(directory, **options):
        return talos.Scan(x_train, y_train, {'c': [.1, 1., 10.]}, local_audit_models._local_audit_iris_model, 'live_data',
            x_val=x_val, y_val=y_val, reduction_method='local_strategy', disable_progress_bar=True,
            experiment_dir=directory, seed=17, reduction_metric='val_loss', minimize_loss=True,
            **initial_flags, **options)
    paused = execute(tmp_path / 'resume', stop_after=2)
    assert paused.status == 'paused'
    checkpoint = CheckpointManager().load(paused.run_dir)
    live = checkpoint['execution_state']['legacy_live_controls']
    assert set(live['data']) == {'x_train'}
    assert live['controls']['print_params'] is True
    assert live['controls']['clear_session'] is False
    resumed = execute(paused.run_dir, resume=True)
    uninterrupted = execute(tmp_path / 'complete')
    assert resumed.status == uninterrupted.status == 'complete'
    np.testing.assert_array_equal(resumed.x_train, x_train + 1)
    np.testing.assert_array_equal(resumed.data.val_loss, uninterrupted.data.val_loss)
    assert resumed.round_history == uninterrupted.round_history
    assert resumed.data._trial_id.tolist() == uninterrupted.data._trial_id.tolist()
    assert resumed._records[0]['split_identity'] != resumed._records[1]['split_identity']
    assert resumed._records[1]['split_identity'] == resumed._records[2]['split_identity']
    artifacts = [item['artifact'] is not None for item in resumed._records]
    assert artifacts == ([True, False, False] if initial_flags['save_weights'] else [False, True, True])
    expected, _ = local_audit_models._local_audit_iris_model(x_train + 1, y_train, x_val, y_val, {'c': 10.})
    assert resumed.data.val_loss.iloc[-1] == expected.history['val_loss'][-1]
    entries = [json.loads(line) for line in (resumed.run_dir / 'audit.jsonl').read_text().splitlines()]
    changes = [item for entry in entries for item in entry['interventions']
               if item.get('op', item.get('operation')) == 'legacy_local_control_change']
    assert changes[0]['changes']['x_train'] == {'before': fingerprint(x_train), 'after': fingerprint(x_train + 1)}
    assert "'c': 1.0" in capsys.readouterr().out


def test_live_data_snapshots_nested_numpy_and_opaque_replacement(tmp_path):
    from talos.reducers.local_strategy import _controls, capture_live_controls, restore_live_controls
    from sklearn.datasets import load_iris
    iris = load_iris()
    original = [iris.data, {'labels': iris.target}]
    scan = SimpleNamespace(model=_local_audit_iris_model, x_train=original,
        _experiment_log=str(tmp_path / 'results.csv'))
    scan._local_controls_baseline = _controls(scan)
    assert capture_live_controls(scan, scan.model) is None
    scan.x_train = (iris.data + 1, {'labels': iris.target})
    state = capture_live_controls(scan, scan.model)
    assert set(state['data']) == {'x_train'}
    restored = SimpleNamespace(model=scan.model, x_train=original, _experiment_log=scan._experiment_log)
    restore_live_controls(restored, decode(json.loads(dumps(state))))
    assert isinstance(restored.x_train, tuple)
    np.testing.assert_array_equal(restored.x_train[0], iris.data + 1)
    np.testing.assert_array_equal(restored.x_train[1]['labels'], iris.target)
    scan.x_train = iter(iris.data)
    scan._local_controls_baseline = _controls(scan)
    scan.x_train = iter(iris.data)
    opaque = capture_live_controls(scan, scan.model)
    assert opaque['data']['x_train']['kind'] == 'nonportable'
    with pytest.raises(NonPortableValueError, match='nonportable live data'):
        restore_live_controls(restored, opaque)


def test_live_dense_torch_tensor_and_parameter_snapshot_preserves_type_dtype_and_grad(tmp_path):
    torch = pytest.importorskip('torch')
    from sklearn.datasets import load_iris
    from talos.reducers.local_strategy import _controls, capture_live_controls, restore_live_controls
    iris = load_iris()
    original = torch.tensor(iris.data, dtype=torch.float64)
    scan = SimpleNamespace(model=_local_audit_iris_model, x_train=original,
        _experiment_log=str(tmp_path / 'results.csv'))
    scan._local_controls_baseline = _controls(scan)
    scan.x_train = [torch.tensor(iris.data + 1, dtype=torch.float64, requires_grad=True),
                    torch.nn.Parameter(torch.tensor(iris.data, dtype=torch.bfloat16))]
    state = capture_live_controls(scan, scan.model)
    restored = SimpleNamespace(model=scan.model, x_train=original, _experiment_log=scan._experiment_log)
    restore_live_controls(restored, decode(json.loads(dumps(state))))
    for source, recovered in zip(scan.x_train, restored.x_train):
        assert type(recovered) is type(source)
        assert recovered.dtype == source.dtype
        assert recovered.requires_grad == source.requires_grad
        torch.testing.assert_close(source, recovered)


def test_live_dense_tensorflow_tensor_and_variable_snapshot_preserves_type_and_dtype(tmp_path):
    tf = pytest.importorskip('tensorflow')
    from sklearn.datasets import load_iris
    from talos.reducers.local_strategy import _controls, capture_live_controls, restore_live_controls
    iris = load_iris()
    original = tf.convert_to_tensor(iris.data, dtype=tf.float64)
    scan = SimpleNamespace(model=_local_audit_iris_model, x_train=original,
        _experiment_log=str(tmp_path / 'results.csv'))
    scan._local_controls_baseline = _controls(scan)
    scan.x_train = [tf.convert_to_tensor(iris.data + 1, dtype=tf.float64),
                    tf.Variable(iris.data, dtype=tf.bfloat16, trainable=False)]
    state = capture_live_controls(scan, scan.model)
    restored = SimpleNamespace(model=scan.model, x_train=original, _experiment_log=scan._experiment_log)
    restore_live_controls(restored, decode(json.loads(dumps(state))))
    for source, recovered in zip(scan.x_train, restored.x_train):
        assert type(recovered) is type(source)
        assert recovered.dtype == source.dtype
        assert getattr(recovered, 'trainable', None) == getattr(source, 'trainable', None)
        np.testing.assert_array_equal(source.numpy(), recovered.numpy())
