"""Current caller domains and original realized rows share one legacy search facade."""
import math
from datetime import datetime

import numpy as np

from talos.experiment.param_domain import ParamDomain, values_equal
from talos.experiment.param_search.legacy_strategy import LegacyStrategy
from talos.experiment.serialization import callable_reference

from ._resume import _BooleanLimit, _constructor_state, _row_state


def _expand_range(values):
    if len(values) != 3:
        raise ValueError('Tuple ranges require (start, end, steps).')
    start, end, steps = values
    if steps <= 0 or start == end:
        raise ValueError('Tuple ranges require positive steps and distinct endpoints.')
    out = np.arange(start, end, (end - start) / steps, dtype=float)
    if isinstance(start, int) and isinstance(end, int):
        out = np.unique(out.astype(int))
    return out


def normalize_domains(params):
    """Expand candidate ranges without materializing their Cartesian product."""
    out = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            out[key] = _expand_range(value).tolist()
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            raise TypeError(f'Parameter {key!r} must be a list or (start, end, steps) tuple.')
    return out


class ParamSpace:
    """Legacy parameter facade over the shared search strategy and queue."""

    def __init__(self, params, param_keys=None, random_method='uniform_mersenne',
                 fraction_limit=None, round_limit=None, time_limit=None,
                 boolean_limit: _BooleanLimit | None = None, seed: int | None = None, **_resume_options: object):
        initial_state = _constructor_state(_resume_options)
        self.params = params
        self.param_keys = list(params) if param_keys is None else list(param_keys)
        self.random_method = random_method
        self.fraction_limit = fraction_limit
        self.round_limit = round_limit
        self.time_limit = time_limit
        self.boolean_limit = boolean_limit
        self.seed = seed
        self.round_counter = 0
        self.shard_namespace, self.shard_id = None, None
        self.p = self._param_input_conversion()
        self._params_temp = [list(self.p[key]) for key in self.param_keys]
        self.dimensions = math.prod(len(values) for values in self._params_temp)
        indices = self._param_apply_limits() if initial_state is None else []
        rows = [self._index_to_values(index) for index in indices]
        if initial_state is None and boolean_limit is not None:
            rows = [values for values in rows if boolean_limit(self._round_parameters_todict(values))]
        self.param_space, self.param_index = _row_state(rows, len(self.param_keys))
        domain = ParamDomain({key: list(self.p[key]) for key in self.param_keys})
        self.strategy = LegacyStrategy(self, domain, seed=seed)
        if initial_state is not None:
            self.strategy.set_state(initial_state)

    def _param_input_conversion(self):
        return normalize_domains({key: self.params[key] for key in self.param_keys})

    def _param_apply_limits(self):
        from talos.reducers.sample_reducer import sample_reducer
        if self.fraction_limit is not None:
            return sample_reducer(self.fraction_limit, self.dimensions, self.random_method, self.seed)
        if self.round_limit is not None:
            return sample_reducer(self.round_limit, self.dimensions, self.random_method, self.seed)
        return range(self.dimensions)

    def _param_range_expansion(self, values):
        return _expand_range(values)

    def _index_to_values(self, index):
        values = []
        for candidates in reversed(self._params_temp):
            index, position = divmod(int(index), len(candidates))
            values.insert(0, candidates[position])
        return values

    def _param_space_creation(self):
        return self.param_space

    def _check_time_limit(self):
        return self.time_limit is None or datetime.strptime(self.time_limit, '%Y-%m-%d %H:%M') > datetime.now()

    def round_parameters(self):
        try:
            return next(self.strategy)
        except StopIteration:
            return False

    def _round_parameters_todict(self, values):
        return dict(zip(self.param_keys, values))

    def _convert_lambda(self, function):
        return function

    def _remove(self, condition, operation, **details):
        if hasattr(self, '_msq'):
            self._msq._log_intervention(operation, source='legacy_reducer', **details)
        self.param_index = [index for index in self.param_index
                            if not condition(self._round_parameters_todict(self.param_space[index]))]

    def remove_is_not(self, label, value):
        self._remove(lambda params: not values_equal(params[label], value), 'keep_is', param=label, value=value)

    def remove_is(self, label, value):
        self._remove(lambda params: values_equal(params[label], value), 'remove_is', param=label, value=value)

    def remove_ge(self, label, value):
        self._remove(lambda params: params[label] >= value, 'remove_ge', param=label, threshold=value)

    def remove_le(self, label, value):
        self._remove(lambda params: params[label] <= value, 'remove_le', param=label, threshold=value)

    def remove_lambda(self, function):
        # Historical callable returns True to keep; never rebuild consumed rows.
        self._remove(lambda params: not function(params), 'legacy_keep_predicate', predicate=callable_reference(function))
