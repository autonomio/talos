from typing import Any
from typing import TypeGuard

import numpy as np
import polars as pl


def _is_integral(value: Any) -> TypeGuard[int | np.integer[Any]]:

    '''
    Check whether a permutation id value is a Python or numpy integer.

    Args:
        value (Any): Candidate permutation id value

    Returns:
        TypeGuard[int | np.integer[Any]]: True if value is an integral number
    '''

    return isinstance(value, (int, np.integer))


def _is_floating(value: Any) -> TypeGuard[float | np.floating[Any]]:

    '''
    Check whether a permutation id value is a Python or numpy float.

    Args:
        value (Any): Candidate permutation id value

    Returns:
        TypeGuard[float | np.floating[Any]]: True if value is a floating number
    '''

    return isinstance(value, (float, np.floating))


def select(context: dict[str, Any],
           *,
           column: str,
           n: object,
           ascending: bool = False) -> list[int | str]:
    '''Return the top n ids from results.csv ordered by one numeric column.'''

    if not isinstance(n, int) or n <= 0:
        raise ValueError('top_n n must be a positive integer')

    results = context.get('results')
    if results is None:
        raise ValueError('top_n selector requires results.csv data in context["results"]')
    if not isinstance(results, pl.DataFrame):
        raise ValueError('top_n context["results"] must be a polars DataFrame')

    missing = [col for col in ('id', column) if col not in results.columns]
    if missing:
        raise ValueError(f'top_n selector input is missing required columns: {missing}')

    def coerce_id(value: Any) -> int | str:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('top_n selector returned a boolean permutation id')
        if value is None or (_is_floating(value) and np.isnan(float(value))):
            raise ValueError('top_n selector returned a missing permutation id')
        if _is_integral(value):
            return int(value)
        if _is_floating(value) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
            if not stripped:
                raise ValueError('top_n selector returned an empty permutation id')
            return stripped

        coerced = str(value).strip()
        if not coerced:
            raise ValueError('top_n selector returned an empty permutation id')
        return coerced

    work = results.select(['id', column]).with_columns(
        pl.col(column).cast(pl.Float64, strict=False)
    )
    work = work.filter(pl.col(column).is_not_null() & pl.col(column).is_not_nan())
    if work.is_empty():
        return []

    sort_keys = [str(coerce_id(value)) for value in work['id'].to_list()]
    work = work.with_columns(pl.Series('_id_sort_key', sort_keys))
    ranked = work.sort([column, '_id_sort_key'], descending=[not ascending, False])

    return [coerce_id(value) for value in ranked.head(n)['id'].to_list()]
