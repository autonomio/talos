"""Original realized legacy rows precede identity checks; checkpoints restore consumption later."""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from talos.experiment import provenance, serialization

__all__ = ['_constructor_state', '_legacy_key_witness', '_resume_initial_state', '_row_state', '_strategy_identity']
_BooleanLimit = Callable[[dict[str, object]], bool]


class _Codec(Protocol):
    """Decoded legacy records cross the untyped serializer boundary as objects."""
    def decode(self, value: object) -> object: ...


class _Provenance(Protocol):
    """Strategy identity retains the existing provenance implementation."""
    def source_identity(self, value: object) -> object: ...


class _StrategyState(Protocol):
    """Search strategies retain their original state representation."""
    def get_state(self) -> dict[str, object]: ...


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError('Saved legacy strategy state must contain objects.')
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError('Saved legacy strategy state must contain string keys.')
    return {str(key): item for key, item in mapping.items()}


def _read_identity(directory: Path) -> dict[str, object]:
    with (directory / 'metadata.json').open(encoding='utf-8') as source:
        raw: object = json.load(source)
    return _mapping(_mapping(raw).get('identity'))


def _resume_initial_state(options: Mapping[str, object], param_keys: list[str]) -> dict[str, object] | None:
    """Recover original realized rows without repeating sampling or predicates."""
    if not options.get('resume') or options.get('search_strategy') is not None:
        return None
    directory = options.get('experiment_dir')
    if not isinstance(directory, (str, Path)):
        raise ValueError('resume requires experiment_dir')
    identity = _read_identity(Path(directory))
    if 'legacy_param_keys' in identity and identity['legacy_param_keys'] != param_keys:
        raise ValueError('Saved dictionary parameter key order differs from the caller.')
    strategy = _mapping(identity.get('strategy'))
    if _mapping(strategy.get('class')).get('qualname') != 'LegacyStrategy':
        raise ValueError('Dictionary Scan resume requires a saved LegacyStrategy.')
    decoded = cast(_Codec, serialization).decode(strategy.get('initial_state'))
    state = _mapping(decoded)
    required = {'param_space', 'pending', 'round_counter', 'generated_count',
                'known_values', 'shard_namespace', 'shard_id'}
    if not required <= state.keys():
        raise ValueError('Saved legacy strategy is missing original row state.')
    raw_rows = state['param_space']
    if not isinstance(raw_rows, np.ndarray):
        raise ValueError('Saved legacy strategy must contain an object row array.')
    rows = cast(NDArray[np.object_], raw_rows)
    if rows.ndim != 2 or rows.dtype != np.dtype(object):
        raise ValueError('Saved legacy strategy must contain a two-dimensional object row array.')
    if state['pending'] != list(range(len(rows))) or state['round_counter'] != 0 or state['generated_count'] != 0:
        raise ValueError('Saved legacy strategy must contain the original unconsumed rows.')
    _mapping(state['known_values'])
    return state


def _constructor_state(options: Mapping[str, object]) -> dict[str, object] | None:
    if options.keys() - {'_initial_state'}:
        raise TypeError('Unexpected ParamSpace keyword argument.')
    state = options.get('_initial_state')
    return None if state is None else _mapping(state)


def _row_state(rows: Sequence[Sequence[object]], columns: int) -> tuple[NDArray[np.object_], list[int]]:
    """Keep realized row positions and their fresh queue indexes aligned."""
    matrix = np.empty((len(rows), columns), dtype=object)
    for index, values in enumerate(rows):
        for column, value in enumerate(values):
            matrix[index, column] = value
    return matrix, list(range(len(rows)))


def _strategy_identity(strategy: _StrategyState) -> dict[str, object]:
    """Preserve the original strategy identity contract during row recovery."""
    state = strategy.get_state().copy()
    state.pop('rng_state', None)
    return {'class': cast(_Provenance, provenance).source_identity(type(strategy)),
            'seed': getattr(strategy, '_seed', None), 'initial_state': state}


def _legacy_key_witness(params: object, directory: Path, resume: bool, legacy: object) -> dict[str, object]:
    """Attest new key order without adding fields to historical identities."""
    if legacy is None or not isinstance(params, dict):
        return {}
    if resume and 'legacy_param_keys' not in _read_identity(directory):
        return {}
    return {'legacy_param_keys': list(cast(dict[object, object], params))}
