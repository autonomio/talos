from functools import wraps
from importlib import import_module
import json
from pathlib import Path
import inspect
from talos.yaml.errors import ValidationError
from talos.yaml.resolver import caller_imports, load_sfd, resolve_values, validate_sfd
from talos.yaml.store import canonical_manifest_id
from talos.yaml.validator import validate


def _caller_scope(function):
    @wraps(function)
    def scoped(self, *args, **kwargs):
        with caller_imports(self._source):
            return function(self, *args, **kwargs)
    return scoped


class CompiledSFD:
    """Compile a generic manifest into the native params/prep/model SFD."""
    def __init__(self, document, source_path=None, *, source_module=None):
        outcome = validate(document)
        if not outcome.valid:
            raise ValidationError(outcome.errors)
        self._yaml = document
        self.source_path = Path(source_path).resolve() if source_path else None
        self.module_reference = document['sfd']['module']
        base = self.source_path.parent if self.source_path else Path.cwd()
        self._source = validate_sfd(source_module, self.module_reference) if source_module is not None else load_sfd(self.module_reference, base)
        supplied_manifest = getattr(self._source, 'manifest', None)
        with caller_imports(self._source):
            self.manifest = supplied_manifest() if callable(supplied_manifest) else supplied_manifest
        self._prep_function = getattr(self._source, 'prep', None) or getattr(self.manifest, 'prepare_data', None)
        self._model_function = getattr(self._source, 'model', None) or getattr(self.manifest, 'run_model', None)
        if not all(callable(function) for function in (self._prep_function, self._model_function)):
            raise TypeError('Caller manifest must expose prepare_data and run_model')
        self._model_source = supplied_manifest if callable(supplied_manifest) else self._model_function
        self._resume_metadata = None
        self.__name__ = self._source.__name__
        self.__file__ = getattr(self._source, '__file__', None)
    @classmethod
    def from_run(cls, run_dir):
        """Compile the exact recorded manifest against verified saved caller sources."""
        from talos.experiment.source_snapshot import hydrate_sources, verify_sources
        raw = json.loads((Path(run_dir) / 'metadata.json').read_text())
        reference = raw['yaml_reference']
        if reference.get('manifest_id') != canonical_manifest_id(reference['content']):
            raise ValueError('Recorded manifest content does not match its hash')
        bundle = verify_sources(raw, run_dir, originals=True)
        source = None
        if bundle and bundle.get('modules'):
            name = raw['sfd']['module']
            if name not in bundle['modules']:
                raise ValueError('Recorded caller module is missing from the source bundle')
            hydrate_sources(raw, run_dir)
            source = import_module(name)
        compiled = cls(reference['content'], source_path=reference.get('source_path'), source_module=source)
        if source is not None:
            compiled._resume_metadata = raw
        return compiled
    @_caller_scope
    def params(self):
        from talos.parameters.ParamSpace import normalize_domains
        values = normalize_domains(dict(self._source.params()))
        values.update(self._yaml['sfd'].get('params', {}))
        for key, domain in values.items():
            if not isinstance(domain, (list, tuple, range)) or not len(domain):
                raise ValueError(f'Parameter {key} needs a nonempty list, tuple or range')
        return resolve_values({key: list(domain) for key, domain in values.items()})
    @_caller_scope
    def prep(self, data=None, round_params=None):
        if data is None:
            data = resolve_values(self._yaml['sfd'].get('context', self._yaml.get('uel', {}).get('context')))
        if isinstance(data, dict) and 'task' in self._yaml['sfd']:
            data = {'task': self._yaml['sfd']['task'], **data}
        function = self._prep_function
        signature = inspect.signature(function)
        choices = [((data, round_params), {}), ((), {'data': data, 'round_params': round_params}),
                   ((data,), {'round_params': round_params}), ((data,), {}),
                   ((), {'round_params': round_params}), ((), {})]
        for args, kwargs in choices:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            return function(*args, **kwargs)
        raise TypeError('Caller prep must accept context and optional round_params')
    @_caller_scope
    def model(self, prepared, round_params):
        return self._model_function(prepared, round_params)
    @_caller_scope
    def pruning_strategies(self):
        return build_pruning_strategies(self._yaml)
    @_caller_scope
    def execute(self, data=None, resume=False, experiment_dir=None, **options):
        from talos.experiment.runner import run
        settings = resolve_values(dict(self._yaml.get('uel', {})))
        settings.pop('search_strategy', None)
        settings.pop('pruning_strategies', None)
        settings.pop('output_path', None)
        settings.pop('context', None)
        settings.pop('n_permutations', None)
        configuration = self._yaml.get('uel', {})
        if 'round_limit' not in settings and 'n_permutations' in configuration:
            settings['round_limit'] = configuration['n_permutations']
        strategy = build_search_strategy(self._yaml, self.params())
        reducers = build_pruning_strategies(self._yaml)
        for key in ('objective', 'backend'):
            if key in self._yaml['sfd']:
                settings.setdefault(key, self._yaml['sfd'][key])
        settings.setdefault('source_model', self._model_source)
        settings.setdefault('source_prep', self._prep_function)
        settings.setdefault('source_params', self._source.params)
        settings.update(options)
        if resume and self._resume_metadata is not None:
            settings['resume_source_metadata'] = self._resume_metadata
        settings.setdefault('experiment_name', self._yaml['metadata']['name'])
        settings['yaml_reference'] = {
            'content': self._yaml,
            'source_path': str(self.source_path) if self.source_path else None,
            'manifest_id': canonical_manifest_id(self._yaml),
        }
        return run(self, data=data, experiment_dir=experiment_dir,
                   search_strategy=strategy, pruning_strategies=reducers,
                   resume=resume, **settings)


def build_search_strategy(document, params=None):
    from talos.experiment.param_domain import ParamDomain
    from talos.experiment.param_search import GridStrategy, RandomStrategy
    domain = ParamDomain(params or document['sfd'].get('params', {}))
    settings = document.get('uel', {}).get('search_strategy', {})
    seed = settings.get('seed', document.get('uel', {}).get('seed'))
    if settings.get('type', 'random') == 'grid':
        return GridStrategy(domain, seed=seed)
    return RandomStrategy(domain, seed=seed)


def build_pruning_strategies(document):
    from talos.experiment.reducer.registry import REDUCER_REGISTRY
    result = []
    for specification in document.get('uel', {}).get('pruning_strategies', []):
        if not isinstance(specification, dict) or specification.get('type') not in REDUCER_REGISTRY:
            raise ValueError(f'Unknown pruning strategy: {specification}')
        result.append(REDUCER_REGISTRY[specification['type']](**resolve_values(specification.get('params', {}))))
    return result


def build_manifest(document, source_path=None):
    return CompiledSFD(document, source_path=source_path)
