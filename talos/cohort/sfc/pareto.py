from typing import Any, cast
from typing import TypeGuard

import numpy as np
import numpy.typing as npt
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
           target_count: object = 20,
           metric_cols: list[str] | None = None,
           directions: dict[str, str] | None = None) -> list[int | str]:
    '''Return a multi-objective Pareto front capped by deterministic rank.'''

    if not isinstance(target_count, int) or target_count <= 0:
        raise ValueError('pareto target_count must be a positive integer')
    if not metric_cols:
        raise ValueError('pareto requires caller metric_cols')
    directions = directions or {}
    if any(direction not in ('min', 'max') for direction in directions.values()):
        raise ValueError('pareto directions must be min or max')

    results = context.get('results')
    if results is None:
        raise ValueError('pareto selector requires results.csv data in context["results"]')
    if not isinstance(results, pl.DataFrame):
        raise ValueError('pareto context["results"] must be a polars DataFrame')

    missing = [col for col in ['id', *metric_cols] if col not in results.columns]
    if missing:
        raise ValueError(f'pareto selector input is missing required columns: {missing}')

    def coerce_id(value: Any) -> int | str:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('pareto selector returned a boolean permutation id')
        if value is None or (_is_floating(value) and np.isnan(float(value))):
            raise ValueError('pareto selector returned a missing permutation id')
        if _is_integral(value):
            return int(value)
        if _is_floating(value) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
            if not stripped:
                raise ValueError('pareto selector returned an empty permutation id')
            return stripped

        coerced = str(value).strip()
        if not coerced:
            raise ValueError('pareto selector returned an empty permutation id')
        return coerced

    work = results.select(['id', *metric_cols]).with_columns(
        [pl.col(col).cast(pl.Float64, strict=False) for col in metric_cols]
    )
    work = work.with_columns([(-pl.col(col)).alias(col) for col in metric_cols
                              if directions.get(col, 'max') == 'min'])

    finite_metrics = cast(npt.NDArray[np.bool_], np.isfinite(work.select(metric_cols).to_numpy()).all(axis=1))
    work = work.filter(finite_metrics)

    if work.is_empty():
        return []

    values = work.select(metric_cols).to_numpy()
    keep = np.ones(len(values), dtype=bool)
    for idx, row in enumerate(values):
        if not keep[idx]:
            continue
        dominates = np.all(values >= row, axis=1) & np.any(values > row, axis=1)
        dominates[idx] = False
        if dominates.any():
            keep[idx] = False

    front = work.filter(keep)
    parts: list[npt.NDArray[np.float64]] = []
    for col in metric_cols:
        col_values = front[col].to_numpy()
        lo = np.nanmin(col_values)
        hi = np.nanmax(col_values)
        if not np.isfinite(lo) or not np.isfinite(hi):
            parts.append(np.zeros(len(col_values), dtype=float))
        elif hi == lo:
            parts.append(np.ones(len(col_values), dtype=float))
        else:
            parts.append((col_values - lo) / (hi - lo))

    scores = np.vstack(parts).mean(axis=0)
    sort_keys = [str(coerce_id(value)) for value in front['id'].to_list()]
    front = front.with_columns([
        pl.Series('_selector_score', scores),
        pl.Series('_id_sort_key', sort_keys),
    ])
    ranked = front.sort(['_selector_score', '_id_sort_key'], descending=[True, False])

    return [coerce_id(value) for value in ranked.head(target_count)['id'].to_list()]
