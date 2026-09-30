"""Generic execution around Limen's forked search, feedback and checkpoint core."""
import copy
import importlib
import importlib.util
import inspect
import json
import signal
import shutil
import sys
import threading
import logging
import numbers
import random
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

from .artifacts import append_round, read_json, read_rounds, truncate_rounds, write_json
from .context import reset_trial_context, set_trial_context
from .provenance import environment, fingerprint, source_identity, file_hash
from .serialization import content_hash, dumps
from .source_snapshot import snapshot_sources, hydrate_sources
from .log_projection import LogQueueView, log_frame

logger = logging.getLogger(__name__)


def load_sfd(sfd):
    if not isinstance(sfd, (str, Path)):
        return sfd
    from talos.yaml.resolver import load_sfd as resolve_sfd
    return resolve_sfd(str(sfd))


def _load_recorded_source(metadata, run_dir):
    source = metadata.get('sfd', {})
    path = source.get('source_snapshot') or source.get('source_path')
    if path and not Path(path).is_absolute():
        path = Path(run_dir) / path
    if path and Path(path).is_file():
        if source.get('source_sha256') and file_hash(path) != source['source_sha256']:
            raise ValueError('Saved model source does not match its recorded hash')
        module_name = source.get('module')
        existing = sys.modules.get(module_name)
        existing_path = getattr(existing, '__file__', None)
        verified_existing = existing_path and Path(existing_path).is_file() and file_hash(existing_path) == source.get('source_sha256')
        if module_name and not verified_existing:
            specification = importlib.util.spec_from_file_location(module_name, path)
            loaded = importlib.util.module_from_spec(specification)
            sys.modules[module_name] = loaded
            sys.path.insert(0, str(Path(path).parent))
            try:
                specification.loader.exec_module(loaded)
            finally:
                sys.path.pop(0)


def _strategy_identity(strategy):
    state = strategy.get_state().copy()
    state.pop('rng_state', None)
    return {'class': source_identity(type(strategy)), 'seed': getattr(strategy, '_seed', None), 'initial_state': state}


def _pruner_identity(reducer):
    configuration = {}
    for key in inspect.signature(type(reducer)).parameters:
        if key == 'start_time':
            continue
        if hasattr(reducer, '_' + key):
            configuration[key] = getattr(reducer, '_' + key)
    return {'class': source_identity(type(reducer)), 'configuration': configuration}


def _seed(value):
    random.seed(value)
    np.random.seed(value)
    if 'tensorflow' in sys.modules:
        sys.modules['tensorflow'].random.set_seed(value)
    if 'torch' in sys.modules:
        sys.modules['torch'].manual_seed(value)


def _invoke(function, data, params, emitted_warnings=None):
    signature = inspect.signature(function)
    choices = [
        ((), {'data': data, 'round_params': params}),
        ((data, params), {}),
        ((data,), {'round_params': params}),
        ((data,), {}),
        ((), {'round_params': params}),
        ((), {}),
    ]
    for args, kwargs in choices:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        if emitted_warnings is None:
            return function(*args, **kwargs)
        with warnings.catch_warnings():
            previous = warnings.showwarning
            def capture(message, category, filename, lineno, file=None, line=None):
                emitted_warnings.append({'message': str(message), 'category': category.__name__, 'filename': filename, 'line': lineno})
                previous(message, category, filename, lineno, file=file, line=line)
            warnings.showwarning = capture
            return function(*args, **kwargs)
    raise TypeError(f'{function.__qualname__} must accept data and round_params')


def _performance_target(value):
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 3 or not isinstance(value[0], str)
            or not isinstance(value[1], numbers.Real) or not isinstance(value[2], bool)):
        raise ValueError('performance_target requires [metric, numeric threshold, minimize_boolean]')
    return value


def _target_reached(metrics, target):
    if target is None:
        return False
    metric, threshold, minimize = target
    if metric not in metrics:
        raise ValueError(f'performance_target metric is missing: {metric}')
    return metrics[metric] <= threshold if minimize else metrics[metric] >= threshold


def _entropy(history):
    from scipy.stats import entropy
    result = {}
    for key, series in history.items():
        if key.startswith('val_'):
            continue
        values = np.asarray(series, dtype=float)
        validation = np.asarray(history.get('val_' + key, []), dtype=float)
        if not values.size or np.any(values < 0) or not np.isfinite(values).all() or values.sum() <= 0:
            result[key] = np.nan
        elif len(validation) == len(values) and np.all(validation >= 0) and np.isfinite(validation).all() and validation.sum() > 0:
            result[key] = float(entropy(validation, values))
        else:
            result[key] = float(entropy(values))
    return result


class RunResult:
    def __init__(self, run_dir, params, experiment_name, objective=None):
        self.run_dir = Path(run_dir).resolve()
        self.params = params
        self.objective = objective
        self.models = []
        self.artifacts = []
        self.saved_models = []
        self.saved_weights = []
        self.round_history = []
        self._rows = []
        self._records = []
        self.parameter_columns = {}
        self.data = pd.DataFrame()
        self.round_times = pd.DataFrame(columns=['start', 'end', 'duration'])
        self.learning_entropy = pd.DataFrame()
        self.details = pd.Series({'experiment_name': experiment_name, 'experiment_id': self.run_dir.name,
                                  'run_dir': str(self.run_dir), 'status': 'running'})
        self.x = None
        self.y = None
        self.metadata = {}
        self.status = 'running'

    def _append(self, record, model=None, serialized_model=None, weights=None):
        self._records.append(record)
        self._rows.append(self._record_row(record))
        self.round_history.append(record['history'])
        descriptor = record.get('artifact')
        if descriptor is not None:
            descriptor = copy.deepcopy(descriptor)
            for key in ('path', 'factory_source'):
                if descriptor.get(key) and not Path(descriptor[key]).is_absolute():
                    descriptor[key] = str(self.run_dir / descriptor[key])
        self.artifacts.append(descriptor)
        self.models.append(model)
        self.saved_models.append(serialized_model)
        self.saved_weights.append(weights)
        self._refresh()

    def _record_row(self, record):
        row = dict(record['row'])
        if 'params' in record:
            for column in record.get('parameter_columns', {}).values():
                row.pop(column, None)
            row.update({self.parameter_columns.get(key, key): value for key, value in record['params'].items()})
        return row

    def _parameter_columns(self, metrics, params):
        reserved = {'start', 'end', 'duration', 'execution_time', 'round_epochs', '_warnings', '_trial_id', '_param_hash'}
        occupied = set(params) | set(metrics) | reserved | set(self.parameter_columns.values())
        for key in params:
            current = self.parameter_columns.get(key, key)
            if current in metrics or current in reserved:
                candidate = 'param__' + key
                while candidate in occupied:
                    candidate = 'param__' + candidate
                self.parameter_columns[key] = candidate
                occupied.add(candidate)
            else:
                self.parameter_columns.setdefault(key, current)
        self._rows = [self._record_row(record) for record in self._records]
        self._refresh()

    def _refresh(self):
        self.data = pd.DataFrame(self._rows)
        if len(self.data):
            self.round_times = self.data[['start', 'end', 'duration']].copy()
            self.learning_entropy = pd.DataFrame([_entropy(h) for h in self.round_history])
        self.details['rounds'] = len(self.data)

    @property
    def experiment_log(self):
        return self.data

    def request_pause(self):
        self._pause_requested = True

    request_shutdown = request_pause

    def best_model(self, metric=None, asc=None, saved=False, custom_objects=None, model_factory=None):
        from talos.utils.best_model import activate_model, best_model
        metric, asc = self._objective(metric, asc)
        return activate_model(self, best_model(self, metric, asc), saved, custom_objects,
                              **({'model_factory': model_factory} if model_factory else {}))

    def _objective(self, metric, asc):
        objective = self.objective or {}
        if isinstance(objective, str):
            objective = {'metric': objective}
        if metric is None:
            metric = objective.get('metric')
            if metric is None:
                metric = next((name for name in ('val_loss', 'loss', 'val_accuracy', 'val_acc', 'accuracy', 'acc', 'score') if name in self.data), None)
            if metric is None:
                raise ValueError('Specify an objective metric for model selection')
        if asc is None:
            asc = objective.get('direction', 'min' if 'loss' in metric or 'mae' in metric else 'max') == 'min'
        return metric, asc

    def predict(self, x, metric=None, asc=None, model_id=None, **kwargs):
        from talos.commands.predict import Predict
        metric, asc = self._objective(metric, asc)
        return Predict(self).predict(x, metric, asc, model_id=model_id, **kwargs)

    def evaluate_models(self, *args, **kwargs):
        from talos.commands.evaluate import evaluate_models
        return evaluate_models(self, *args, **kwargs)

    @classmethod
    def load(cls, run_dir, custom_objects=None, model_factory=None):
        run_dir = Path(run_dir).resolve()
        raw = json.loads((run_dir / 'metadata.json').read_text())
        if raw.get('source_bundle'):
            hydrate_sources(raw, run_dir)
        else:
            _load_recorded_source(raw, run_dir)
        metadata = read_json(run_dir / 'metadata.json')
        result = cls(run_dir, metadata['params'], metadata['experiment_name'], metadata.get('objective'))
        result.parameter_columns = dict(metadata.get('parameter_columns', {}))
        checkpoint = read_json(run_dir / 'checkpoint.json')
        execution_state = checkpoint.get('execution_state', {})
        result.parameter_columns.update(execution_state.get('parameter_columns', {}))
        count = checkpoint['metadata']['experiment_round'] + 1
        records = read_rounds(run_dir / 'round_data.jsonl', count)
        if len(records) != count:
            raise ValueError('Checkpoint references missing completed trial artifacts')
        if records:
            result.parameter_columns.update(records[-1].get('parameter_columns', {}))
        for record in records:
            result._append(record)
        result.details = pd.Series({**metadata['details'], **execution_state.get('details', {})})
        result.details['rounds'] = count
        result.metadata = metadata
        result.status = result.details.get('status', 'unknown')
        result.custom_objects = custom_objects
        result.model_factory = model_factory
        return result


def run(sfd, data=None, *, params=None, experiment_name='experiment', experiment_dir=None,
        search_strategy=None, pruning_strategies=None, feedback_interval=100, checkpoint_interval=1,
        seed=None, round_limit=None, resume=False, progress_bar=True, objective=None,
        backend=None, model_factory=None, **options):
    from talos.backends import backend_for, normalise_result
    from talos.experiment.checkpoint_manager import CheckpointManager
    from talos.experiment.feedback_controller import FeedbackController
    from talos.experiment.msq import MSQ
    from talos.experiment.param_domain import ParamDomain
    from talos.experiment.param_search.grid_strategy import GridStrategy
    from talos.experiment.param_search.random_strategy import RandomStrategy

    sfd = load_sfd(sfd)
    backend = backend or getattr(sfd, 'backend', None)
    if seed is not None and backend in ('torch', 'pytorch', 'tensorflow', 'tf', 'tf.keras', 'keras'):
        importlib.import_module({'pytorch': 'torch', 'tf': 'tensorflow', 'tf.keras': 'tensorflow'}.get(backend, backend))
    if seed is not None:
        _seed(int(seed) % (2 ** 32))
    model_function = getattr(sfd, 'model', None)
    prep_function = getattr(sfd, 'prep', None)
    manifest = getattr(sfd, 'manifest', None)
    manifest_factory = manifest if callable(manifest) else None
    if manifest_factory is not None:
        manifest = manifest_factory()
    if manifest is not None:
        model_function = model_function or getattr(manifest, 'run_model', None)
        prep_function = prep_function or getattr(manifest, 'prepare_data', None)
    if not callable(model_function):
        raise TypeError('SFD must expose a callable model')
    params_function = getattr(sfd, 'params', None)
    if params is None:
        if not callable(params_function):
            raise TypeError('SFD must expose params() or receive params explicitly')
        params = params_function()
    legacy = options.get('legacy_context')
    performance_target = _performance_target(options.get('performance_target')) if legacy is None else None
    if search_strategy is None:
        if legacy is not None:
            search_strategy = legacy.param_object.strategy
        else:
            from talos.parameters.ParamSpace import normalize_domains
            domain = ParamDomain(normalize_domains(params))
            if options.get('random_search', False):
                search_strategy = RandomStrategy(domain, seed=seed)
            else:
                search_strategy = GridStrategy(domain, seed=seed)
    domain = search_strategy.domain
    limit = round_limit
    if limit is None:
        limit = options.get('n_permutations')
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError('round_limit must be a positive integer')
    if not isinstance(feedback_interval, int) or feedback_interval < 1:
        raise ValueError('feedback_interval must be a positive integer')
    checkpoint_manager = CheckpointManager(checkpoint_interval=checkpoint_interval)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    output_format = options.get('output_format', 'csv')
    if output_format not in ('csv', 'parquet'):
        raise ValueError('output_format must be csv or parquet')
    save_models = options.get('save_models', legacy is None)
    save_weights = options.get('save_weights', True)
    clear_session = options.get('clear_session', True)
    keep_models = options.get('retain_models', not save_models and save_weights)
    if experiment_dir is None:
        if resume:
            raise ValueError('resume requires experiment_dir')
        identifier = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:8]
        run_dir = Path(experiment_name) / identifier
    else:
        run_dir = Path(experiment_dir)
    run_dir = run_dir.resolve()
    if not resume and any(run_dir.glob('*')):
        raise FileExistsError(f'Experiment directory is not empty: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    result = RunResult(run_dir, params, experiment_name, objective)
    yaml_reference = options.get('yaml_reference')
    source_model = options.get('source_model', manifest_factory or model_function)
    source_prep = options.get('source_prep', prep_function)
    source_params = options.get('source_params', params_function)
    source_sfd = source_params if callable(source_params) and (getattr(source_model, '__module__', None) or '').startswith('talos.') else source_model
    recorded_sources = options.get('resume_source_metadata')
    previous_sources = None
    if recorded_sources is not None:
        if not resume:
            raise ValueError('Recorded source identities require resume')
        previous_raw = json.loads((run_dir / 'metadata.json').read_text())
        if recorded_sources.get('identity_hash') != previous_raw.get('identity_hash'):
            raise ValueError('Recorded source metadata does not match this run')
        for name, entry in previous_raw.get('source_bundle', {}).get('modules', {}).items():
            original = Path(entry['original_path'])
            if not entry.get('namespace') and original.is_file() and file_hash(original) != entry['sha256']:
                raise ValueError(f'Caller source changed since its snapshot: {name}')
        hydrate_sources(previous_raw, run_dir)
        previous_sources = read_json(run_dir / 'metadata.json')
    source_callables = manifest.source_functions() if manifest is not None and hasattr(manifest, 'source_functions') else []
    source_bundle = snapshot_sources(source_model, source_prep,
                                    [params if isinstance(params, dict) else domain.params, source_params, source_callables], run_dir)
    model_identity, prep_identity = source_identity(source_model), source_identity(source_prep)
    if previous_sources is not None:
        recorded_bundle = previous_sources['source_bundle']
        module_hashes = lambda bundle: {name: entry.get('sha256') for name, entry in bundle['modules'].items()}
        if module_hashes(source_bundle) != module_hashes(recorded_bundle):
            raise ValueError('Caller source module graph changed since checkpoint')
        for key, current in (('model', model_identity), ('prep', prep_identity)):
            expected = previous_sources['identity'][key]
            if current is None and expected is None:
                continue
            if current is None or expected is None or not expected.get('source_sha256'):
                raise ValueError('Recorded caller sources require verified source files')
            if any(current.get(field) != expected.get(field) for field in ('module', 'qualname', 'source_sha256')):
                raise ValueError(f'Caller {key} source changed since checkpoint')
        model_identity, prep_identity = previous_sources['identity']['model'], previous_sources['identity']['prep']
        source_bundle = recorded_bundle
    identity = {'params': params if isinstance(params, dict) else domain.params,
                'model': model_identity, 'prep': prep_identity,
                'data': options.get('data_fingerprint', fingerprint(data)), 'seed': seed,
                'source_modules': {name: entry.get('sha256') for name, entry in source_bundle.get('modules', {}).items()},
                'strategy': _strategy_identity(search_strategy), 'manifest': yaml_reference.get('content') if yaml_reference else None,
                'objective': objective, 'performance_target': performance_target, 'output_format': output_format, 'save_models': save_models, 'save_weights': save_weights, 'limit': limit,
                'pruners': [_pruner_identity(r) for r in (pruning_strategies or [])],
                'preparation_manifest': manifest.configuration() if manifest is not None and hasattr(manifest, 'configuration') else None,
                'feedback_interval': feedback_interval, 'intra_callback': source_identity(options.get('intra_callback')),
                'prep_each_round': options.get('prep_each_round', legacy is None),
                'legacy_controls': {k: source_identity(getattr(legacy, k)) if callable(getattr(legacy, k)) else getattr(legacy, k)
                                    for k in ('reduction_method', 'reduction_interval', 'reduction_window',
                                              'reduction_threshold', 'reduction_metric', 'minimize_loss', 'performance_target')} if legacy else None}
    identity_hash = content_hash(identity)
    current_environment = environment()
    msq = MSQ(search_strategy, domain, n_permutations=limit)
    reducers = pruning_strategies or []
    feedback = FeedbackController(feedback_interval=feedback_interval, pruning_strategies=reducers,
                                  intra_callback=options.get('intra_callback'),
                                  intervention_path=run_dir / 'interventions.json', audit_log_path=run_dir / 'audit.jsonl')
    result.queue, result.domain, result.search_strategy, result.feedback = msq, domain, search_strategy, feedback
    result.details['seed'] = seed
    result.details['identity_hash'] = identity_hash
    metadata = {'schema_version': '1.0', 'experiment_name': experiment_name, 'params': params if isinstance(params, dict) else domain.params,
                'identity': identity, 'identity_hash': identity_hash, 'environment': current_environment,
                'objective': objective, 'yaml_reference': yaml_reference, 'sfd': previous_sources['sfd'] if previous_sources is not None else source_identity(source_sfd),
                'source_bundle': source_bundle, 'parameter_columns': result.parameter_columns,
                'determinism': {'seed': seed, 'backend_deterministic_ops': 'caller-controlled'},
                'details': result.details.to_dict()}
    if metadata['sfd'].get('source_path'):
        snapshot = run_dir / 'sources' / (metadata['sfd']['source_sha256'][:12] + '_' + Path(metadata['sfd']['source_path']).name)
        if not resume:
            snapshot.parent.mkdir(exist_ok=True)
            shutil.copyfile(metadata['sfd']['source_path'], snapshot)
        metadata['sfd']['source_snapshot'] = str(snapshot.relative_to(run_dir))
    result.metadata = metadata
    round_path = run_dir / 'round_data.jsonl'
    completed = 0
    execution_state = {}
    if resume:
        previous = read_json(run_dir / 'metadata.json')
        if previous['environment'] != current_environment:
            raise ValueError('Environment changed since checkpoint; use the recorded backend/dependency versions')
        state = checkpoint_manager.validate(run_dir, content_hash=identity_hash, strategy_type=type(search_strategy).__name__)
        execution_state = state.get('execution_state', {})
        domain.set_state(state['domain_state'])
        msq.set_state(state['msq_state'])
        if 'feedback_controller_state' in state:
            feedback.set_state(state['feedback_controller_state'])
        for reducer, reducer_state in zip(reducers, state.get('pruning_strategy_states', []), strict=True):
            reducer.set_state(reducer_state)
        completed = state['metadata']['experiment_round'] + 1
        records = read_rounds(round_path, completed)
        if len(records) != completed:
            raise ValueError('Checkpoint references missing completed trial artifacts')
        truncate_rounds(round_path, completed)
        result.parameter_columns = {**previous.get('parameter_columns', {}), **execution_state.get('parameter_columns', {})}
        if records:
            result.parameter_columns.update(records[-1].get('parameter_columns', {}))
        for record in records:
            result._append(record)
    else:
        write_json(run_dir / 'metadata.json', metadata)
        round_path.touch()

    def save_checkpoint():
        live_controls = None
        if legacy is not None:
            for key in ('random_method', 'reduction_method', 'reduction_metric', 'reduction_interval',
                        'reduction_window', 'reduction_threshold', 'minimize_loss'):
                value = getattr(legacy, key)
                result.details[key] = source_identity(value) if callable(value) else value
            result.details['x_shape'] = getattr(legacy.x, 'shape', 'multi-input')
            result.details['y_shape'] = getattr(legacy.y, 'shape', 'multi-output')
        if legacy is not None:
            from talos.reducers.local_strategy import capture_live_controls
            live_controls = capture_live_controls(legacy, source_model)
        saved_execution = {'parameter_columns': dict(result.parameter_columns), 'details': result.details.to_dict(),
                           'legacy_live_controls': live_controls}
        checkpoint_manager.save(run_dir, msq, domain, completed - 1, limit if limit is not None else domain.total_combinations,
                                strategy_type=type(search_strategy).__name__, content_hash=identity_hash,
                                feedback_controller=feedback, pruning_strategies=reducers, execution_state=saved_execution)
        metadata['parameter_columns'] = result.parameter_columns
        metadata['details'] = result.details.to_dict()
        write_json(run_dir / 'metadata.json', metadata)

    bar = tqdm(total=limit or domain.total_combinations, initial=completed, disable=not progress_bar)
    if legacy is not None:
        legacy.pbar = bar
        legacy._experiment_log = str(run_dir / 'results.csv')
        legacy._experiment_id = run_dir.name
        legacy._saved_models_path = str(run_dir / 'models')
    csv_columns = list(result.data.columns)
    if resume:
        result.data.to_csv(run_dir / 'results.csv', index=False)
    result.status = 'running'
    event_callback = options.get('event_callback')
    deadline = options.get('time_limit')
    if isinstance(deadline, str):
        deadline = datetime.strptime(deadline, '%Y-%m-%d %H:%M').timestamp()
    sentinel = object()
    prepared_cache = sentinel
    split_identity_cache = None
    prep_each_round = options.get('prep_each_round', legacy is None)
    saved_handlers = {}
    if threading.current_thread() is threading.main_thread():
        def stop_signal(signum, frame):
            raise KeyboardInterrupt
        for signum in (signal.SIGINT, signal.SIGTERM):
            saved_handlers[signum] = signal.signal(signum, stop_signal)
    caller_path = str(run_dir / source_bundle['directory'])
    sys.path.insert(0, caller_path)
    checkpoint_safe = not resume
    try:
        if legacy is not None and execution_state.get('legacy_live_controls') is not None:
            from talos.reducers.local_strategy import restore_live_controls
            restore_live_controls(legacy, execution_state['legacy_live_controls'])
        if resume and completed and prep_function:
            first = result._rows[0]
            first_params = records[0].get('params', {key: first[result.parameter_columns.get(key, key)] for key in domain.params})
            if seed is not None:
                _seed(int(seed) % (2 ** 32))
            replay_prepared = _invoke(prep_function, data, first_params)
            first_record = records[0]
            if fingerprint(replay_prepared) != first_record['split_identity']:
                raise ValueError('Prepared data or split changed since checkpoint')
            if not prep_each_round:
                prepared_cache, split_identity_cache = replay_prepared, first_record['split_identity']
        checkpoint_safe = True
        save_checkpoint()
        if performance_target is not None and result._records and _target_reached(result._records[-1]['row'], performance_target):
            result.status = 'complete'
        while result.status == 'running' and (deadline is None or time.time() < deadline):
            if getattr(result, '_pause_requested', False) or options.get('pause_requested', lambda: False)():
                result.status = 'paused'
                break
            changes = feedback._collect_from_file()
            for intervention in changes:
                feedback._apply_intervention(msq, intervention)
            if changes:
                feedback._write_audit_entry(completed, changes, [], msq)
            queue_state = msq.get_state()
            trial_index = completed
            column_state = dict(result.parameter_columns)
            trial_committed = False
            domain_state = domain.get_state()
            try:
                combination = next(msq)
            except StopIteration:
                break
            metadata_keys = {'_id', '_trial_id', '_round_index', '_injected', '_generation_index', '_search_strategy', '_param_hash'}
            round_params = {key: value for key, value in combination.items() if key not in metadata_keys}
            trial_id = combination.get('_trial_id', f'{combination["_id"]}:{completed}')
            token = set_trial_context({'run_dir': str(run_dir), 'experiment_name': experiment_name,
                                       'experiment_id': run_dir.name, 'trial_id': trial_id, 'model_id': completed,
                                       'params': round_params})
            adapter = None
            model = None
            start_clock = time.perf_counter()
            started = datetime.now(timezone.utc).isoformat()
            def rollback_trial():
                nonlocal completed
                domain.set_state(domain_state)
                msq.set_state(queue_state)
                completed = trial_index
                result.parameter_columns = column_state
                for attribute in ('_records', 'round_history', 'models', 'artifacts', 'saved_models', 'saved_weights'):
                    setattr(result, attribute, getattr(result, attribute)[:trial_index])
                result._rows = [result._record_row(record) for record in result._records]
                truncate_rounds(round_path, trial_index)
                result._refresh()

            try:
                emitted_warnings = []
                if legacy is not None:
                    data.update({key: getattr(legacy, key) for key in ('x_train', 'y_train', 'x_val', 'y_val')})
                trial_print_params = options.get('print_params', False) if legacy is None else legacy.print_params
                if trial_print_params:
                    print(round_params)
                trial_seed = None if seed is None else (int(seed) + completed) % (2 ** 32)
                if trial_seed is not None:
                    _seed(trial_seed)
                if prep_each_round or prepared_cache is sentinel:
                    prepared = _invoke(prep_function, data, round_params, emitted_warnings) if prep_function else data
                    split_identity = fingerprint(prepared)
                    if not prep_each_round:
                        prepared_cache, split_identity_cache = prepared, split_identity
                else:
                    prepared, split_identity = prepared_cache, split_identity_cache
                    if legacy is not None:
                        split_identity = fingerprint(prepared)
                if event_callback:
                    event_callback('trial_started', result, round_params)
                if legacy is not None:
                    data.update({key: getattr(legacy, key) for key in ('x_train', 'y_train', 'x_val', 'y_val')})
                    split_identity = fingerprint(data)
                output = _invoke(model_function, prepared, round_params, emitted_warnings)
                trial_save_models = save_models if legacy is None else legacy.save_models
                trial_save_weights = save_weights if legacy is None else legacy.save_weights
                trial_keep_models = options.get('retain_models', not trial_save_models and trial_save_weights) if legacy is not None else keep_models
                normalized = normalise_result(output, backend=backend, model_factory=model_factory)
                metrics = normalized['metrics']
                reached_target = _target_reached(metrics, performance_target)
                collisions = set(metrics) & {'start', 'end', 'duration', 'execution_time', 'round_epochs', '_warnings', '_trial_id', '_param_hash'}
                if collisions:
                    raise ValueError(f'Metric names collide with result metadata: {sorted(collisions)}')
                result._parameter_columns(metrics, round_params)
                history = normalized['history']
                model = normalized['model']
                duration = time.perf_counter() - start_clock
                row = {'start': started, 'end': datetime.now(timezone.utc).isoformat(), 'duration': duration, 'execution_time': duration,
                       'round_epochs': max((len(values) for values in history.values()), default=0), '_warnings': dumps(emitted_warnings),
                       **metrics, **{result.parameter_columns[key]: value for key, value in round_params.items()}, '_trial_id': trial_id, '_param_hash': combination['_id']}
                descriptor = None
                serialized_model = None
                weights = None
                if model is not None:
                    adapter = backend_for(model, backend=normalized.get('backend', backend))
                    if trial_save_models or trial_save_weights:
                        descriptor = adapter.save(model, run_dir / 'models' / str(completed),
                                                  model_factory=normalized.get('factory', model_factory))
                        descriptor = copy.deepcopy(descriptor)
                        if descriptor.get('path'):
                            descriptor['sha256'] = file_hash(descriptor['path'])
                            descriptor['path'] = str(Path(descriptor['path']).resolve().relative_to(run_dir))
                        factory_source = descriptor.get('factory_source')
                        if factory_source:
                            source_target = run_dir / 'sources' / (file_hash(factory_source)[:12] + '_' + Path(factory_source).name)
                            source_target.parent.mkdir(exist_ok=True)
                            if not source_target.exists():
                                shutil.copyfile(factory_source, source_target)
                            descriptor['factory_source'] = str(source_target.relative_to(run_dir))
                            descriptor['factory_source_sha256'] = file_hash(source_target)
                    if not trial_save_models and trial_save_weights:
                        if hasattr(model, 'to_json') and hasattr(model, 'get_weights'):
                            serialized_model = model.to_json()
                            weights = model.get_weights()
                        elif hasattr(model, 'state_dict'):
                            weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                            serialized_model = weights
                record = {'schema_version': '1.0', 'experiment_id': run_dir.name,
                          'manifest_id': yaml_reference.get('manifest_id') if yaml_reference else None,
                          'identity_hash': identity_hash, 'round_index': completed, 'trial_id': trial_id,
                          'row': row, 'params': dict(round_params), 'parameter_columns': dict(result.parameter_columns),
                          'history': history, 'warnings': emitted_warnings, 'artifact': descriptor,
                          'split_identity': split_identity,
                          'artifact_policy': {'save_models': trial_save_models, 'save_weights': trial_save_weights, 'clear_session': clear_session if legacy is None else legacy.clear_session},
                          'predictions': normalized.get('predictions'),
                          'backend': normalized.get('backend'), 'seed': None if seed is None else trial_seed, 'status': 'completed'}
                append_round(round_path, record)
                result._append(record, model if trial_keep_models else None, serialized_model, weights)
                completed += 1
                trial_committed = True
                csv_path = run_dir / 'results.csv'
                if completed == 1 or list(result.data.columns) != csv_columns:
                    temporary_csv = run_dir / 'results.csv.tmp'
                    result.data.to_csv(temporary_csv, index=False)
                    temporary_csv.replace(csv_path)
                    csv_columns = list(result.data.columns)
                else:
                    result.data.iloc[-1:].to_csv(csv_path, index=False, header=False, mode='a')
                if legacy is not None:
                    legacy.parameter_columns = result.parameter_columns
                    legacy.round_params = round_params
                    legacy.model_history = SimpleNamespace(history=history)
                    legacy.round_model = model
                    legacy.round_history = result.round_history
                    legacy.data = result.data
                    legacy.result = [list(result.data.columns)] + result.data.values.tolist()
                    legacy._metric_keys = [key for key in history if not key.startswith('val_')]
                    legacy._val_keys = [key for key in history if key.startswith('val_')]
                    from talos.reducers.reduce_run import reduce_run
                    audit_start = len(msq.intervention_log)
                    reduce_run(legacy)
                    changes = msq.intervention_log[audit_start:]
                    if changes:
                        feedback._write_audit_entry(completed, changes, [], msq)
                if feedback.should_trigger(completed):
                    numeric, encoded_columns = log_frame(result)
                    feedback.trigger(numeric, LogQueueView(msq, result.parameter_columns, encoded_columns), search_strategy, completed)
                elif (run_dir / 'interventions.json').exists():
                    changes = feedback._collect_from_file()
                    for intervention in changes:
                        feedback._apply_intervention(msq, intervention)
                    if changes:
                        feedback._write_audit_entry(completed, changes, [], msq)
                result.details['rounds'] = completed
                if checkpoint_manager.should_checkpoint(completed):
                    save_checkpoint()
                bar.update(1)
                if event_callback:
                    event_callback('trial_completed', result, round_params)
                if reached_target:
                    result.status = 'complete'
                    break
                if options.get('stop_after') is not None and completed >= options['stop_after']:
                    result.status = 'paused'
                    break
            except KeyboardInterrupt:
                if not trial_committed:
                    rollback_trial()
                result.status = 'paused'
                break
            except Exception:
                if not trial_committed:
                    rollback_trial()
                result.status = 'failed'
                result.details['status'] = 'failed'
                save_checkpoint()
                raise
            finally:
                reset_trial_context(token)
                trial_clear_session = clear_session if legacy is None else legacy.clear_session
                if trial_clear_session and adapter is not None:
                    adapter.cleanup()
        if result.status == 'running':
            result.status = 'paused' if deadline is not None and time.time() >= deadline else 'complete'
    except BaseException:
        if result.status == 'running':
            result.status = 'failed'
        raise
    finally:
        sys.path.remove(caller_path)
        for signum, handler in saved_handlers.items():
            signal.signal(signum, handler)
        bar.close()
        result.details['status'] = result.status
        result.details['complete_time'] = datetime.now(timezone.utc).isoformat()
        if checkpoint_safe:
            save_checkpoint()
        if checkpoint_safe and output_format == 'parquet':
            frame, _ = log_frame(result)
            temporary_parquet = run_dir / 'results.parquet.tmp'
            frame.write_parquet(temporary_parquet)
            temporary_parquet.replace(run_dir / 'results.parquet')
    result._refresh()
    return result
