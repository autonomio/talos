'''
Declarative filter type constants and builders for named filters.

Each constant defines a filter_type string used in intervention dicts.
FILTER_BUILDERS maps these to factory functions that create filter callables
from filter_params dicts. Keys are extracted eagerly so missing params
fail at filter creation time, not during MSQ iteration.

All builders return a callable with signature (combo: dict) -> bool.
True means REMOVE the combo, False means keep it. This matches
MSQ._passes_filters which treats True as a rejection.

'''

import hashlib
from talos.experiment.serialization import dumps
from talos.experiment.param_domain import values_equal
from collections.abc import Callable
from typing import Any

FILTER_EXCLUDE_VALUE = 'exclude_value'
FILTER_KEEP_VALUES = 'keep_values'
FILTER_KEEP_BETWEEN = 'keep_between'
FILTER_SAMPLE = 'sample'


def _build_exclude_value(fp: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:

    param, value = fp['param'], fp['value']
    return lambda c: values_equal(c[param], value)


def _build_keep_values(fp: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:

    param, values = fp['param'], list(fp['values'])
    return lambda c: not any(values_equal(c[param], value) for value in values)


def _build_keep_between(fp: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:

    param, lower, upper = fp['param'], fp['lower'], fp['upper']
    def outside(c):
        try:
            return not bool(lower <= c[param] <= upper)
        except (TypeError, ValueError):
            return True
    return outside


def _build_sample(fp: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:

    param, value, fraction = fp['param'], fp['value'], fp['fraction']
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"FILTER_SAMPLE fraction must be between 0.0 and 1.0, got {fraction}"
        )
    threshold = round(fraction * 10_000)
    return lambda c: (
        values_equal(c.get(param), value)
        and int.from_bytes(hashlib.blake2b(
            dumps({key: val for key, val in c.items() if key not in
                   {'_id', '_trial_id', '_param_hash', '_round_index', '_injected', '_generation_index', '_search_strategy'}}).encode(), digest_size=8,
        ).digest(), 'big') % 10_000 >= threshold
    )


FILTER_BUILDERS: dict[str, Callable[[dict[str, Any]], Callable[[dict[str, Any]], bool]]] = {
    FILTER_EXCLUDE_VALUE: _build_exclude_value,
    FILTER_KEEP_VALUES: _build_keep_values,
    FILTER_KEEP_BETWEEN: _build_keep_between,
    FILTER_SAMPLE: _build_sample,
}
