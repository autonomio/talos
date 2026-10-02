"""Validate SFD selection and experiment controls before caller imports."""
from collections.abc import Mapping, Sequence
from typing import cast

from talos.yaml.errors import YAMLError


def objective_errors(value: object, path: str) -> list[YAMLError]:
    """Reject ambiguous objective shapes and invalid metric directions."""
    if value is None or (isinstance(value, str) and value):
        return []
    if not isinstance(value, dict):
        return [YAMLError('Expected a metric name or objective mapping', path=path)]
    objective = cast(Mapping[str, object], value)
    errors: list[YAMLError] = []
    metric = objective.get('metric')
    if metric is not None and (not isinstance(metric, str) or not metric):
        errors.append(YAMLError('Must be a nonempty metric name', path=f'{path}.metric'))
    if 'direction' in objective and objective['direction'] not in ('min', 'max'):
        errors.append(YAMLError('Expected min or max', path=f'{path}.direction'))
    return errors


def sfd_errors(value: object) -> list[YAMLError]:
    """Check caller-module fields and parameter domains without executing code."""
    if not isinstance(value, dict):
        return [YAMLError('Must be a mapping', path='sfd')]
    sfd = cast(Mapping[str, object], value)
    errors: list[YAMLError] = []
    if not isinstance(sfd.get('module'), str) or not sfd.get('module'):
        errors.append(YAMLError('Required caller SFD module name or project .py path', path='sfd.module'))
    for key in set(sfd) - {'module', 'params', 'task', 'objective', 'backend', 'context'}:
        errors.append(YAMLError('Unknown SFD field; data acquisition belongs in caller prep/entrypoint', path=f'sfd.{key}'))
    params = sfd.get('params', {})
    if not isinstance(params, dict):
        errors.append(YAMLError('Must be a parameter mapping', path='sfd.params'))
    else:
        for key, values in cast(Mapping[str, object], params).items():
            if not isinstance(values, (list, tuple, range)) or not len(cast(Sequence[object], values)):
                errors.append(YAMLError('Each parameter requires a nonempty sequence', path=f'sfd.params.{key}'))
    return errors + objective_errors(sfd.get('objective'), 'sfd.objective')


def uel_errors(value: object) -> list[YAMLError]:
    """Check sweep limits and boolean policies before their runtime interpretation."""
    if not isinstance(value, dict):
        return [YAMLError('Must be a mapping', path='uel')]
    uel = cast(Mapping[str, object], value)
    errors: list[YAMLError] = []
    allowed = {'n_permutations', 'round_limit', 'seed', 'search_strategy', 'pruning_strategies', 'feedback_interval', 'checkpoint_interval', 'output_format', 'output_path', 'prep_each_round', 'intra_callback', 'objective', 'backend', 'progress_bar', 'time_limit', 'performance_target', 'save_models', 'context'}
    for key in set(uel) - allowed:
        errors.append(YAMLError('Unknown experiment setting', path=f'uel.{key}'))
    for key in ('n_permutations', 'round_limit', 'feedback_interval', 'checkpoint_interval'):
        entry = uel.get(key)
        if key in uel and (isinstance(entry, bool) or not isinstance(entry, int) or entry < 1):
            errors.append(YAMLError('Must be a positive integer', path=f'uel.{key}'))
    for key in ('save_models', 'prep_each_round', 'progress_bar'):
        if key in uel and not isinstance(uel[key], bool):
            errors.append(YAMLError('Must be a boolean', path=f'uel.{key}'))
    errors.extend(search_errors(uel.get('search_strategy', {})))
    if uel.get('output_format', 'csv') not in ('csv', 'parquet'):
        errors.append(YAMLError('Expected csv or parquet', path='uel.output_format'))
    if not isinstance(uel.get('pruning_strategies', []), list):
        errors.append(YAMLError('Must be a list', path='uel.pruning_strategies'))
    return errors + objective_errors(uel.get('objective'), 'uel.objective')


def search_errors(value: object) -> list[YAMLError]:
    """Check the existing grid/random strategy discriminator safely."""
    if isinstance(value, dict) and cast(Mapping[str, object], value).get('type', 'random') in ('random', 'grid'):
        return []
    return [YAMLError('Expected random or grid strategy mapping', path='uel.search_strategy')]
