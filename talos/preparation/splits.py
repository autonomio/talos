import polars as pl

from collections.abc import Sequence
from datetime import date, datetime
from itertools import accumulate
from typing import Any
import math
import numbers


def split_sequential(data: pl.DataFrame, ratios: Sequence[int]) -> list[pl.DataFrame]:

    '''
    Compute sequential data splits with proportional lengths based on ratios.

    Args:
        data (pl.DataFrame): Polars DataFrame to split sequentially
        ratios (Sequence[int]): Sequence of positive integers defining split proportions

    Returns:
        List[pl.DataFrame]: List of DataFrames partitioned sequentially without losing or duplicating rows
    '''

    _validate_ratios(ratios)
    total = data.height
    if total == 0:
        return [data.head(0) for _ in ratios]

    total_ratio = sum(ratios)

    sizes: list[int] = []
    cumulative = 0

    for r in ratios[:-1]:
        chunk_size = int(total * r / total_ratio)
        sizes.append(chunk_size)
        cumulative += chunk_size

    sizes.append(total - cumulative)

    out: list[pl.DataFrame] = []
    start = 0
    for size in sizes:
        out.append(data.slice(start, size))
        start += size

    return out


def split_random(data: pl.DataFrame, ratios: Sequence[int], seed: int | None = None) -> list[pl.DataFrame]:

    '''
    Compute random data splits with proportional lengths based on ratios.

    Args:
        data (pl.DataFrame): Polars DataFrame to split randomly
        ratios (Sequence[int]): Sequence of positive integers defining split proportions
        seed (int): Seed for random number generator

    Returns:
        List[pl.DataFrame]: List of randomly shuffled DataFrames with proportional sizes
    '''

    _validate_ratios(ratios)
    total = data.height
    total_ratio = sum(ratios)
    bounds = [int(total * c / total_ratio) for c in accumulate(ratios)]
    starts = [0, *bounds[:-1]]

    shuffled = data.sample(fraction=1.0, seed=seed, shuffle=True)
    return [shuffled.slice(start, end - start) for start, end in zip(starts, bounds, strict=True)]


def split_by_dates(
    data: pl.DataFrame,
    train_start: date | Any, train_end: date | Any,
    val_start: date | Any, val_end: date | Any,
    test_start: date | Any, test_end: date | Any,
    *, time_col: str = 'datetime',
) -> list[pl.DataFrame]:

    '''
    Split a datetime-indexed DataFrame into train/val/test by half-open
    date windows `[start, end)`.

    Each window selects its rows independently. No row from outside all
    three windows enters any split. Windows must be ordered and non-overlapping.
    Callers choose the datetime column and acquire the data themselves.

    Args:
        data (pl.DataFrame): Input data; must have a `datetime` column
        train_start (date | datetime): Train window start (inclusive)
        train_end   (date | datetime): Train window end (exclusive)
        val_start   (date | datetime): Val window start (inclusive)
        val_end     (date | datetime): Val window end (exclusive)
        test_start  (date | datetime): Test window start (inclusive)
        test_end    (date | datetime): Test window end (exclusive)

    Returns:
        list[pl.DataFrame]: three DataFrames in train, val, test order

    Raises:
        TypeError: If any bound is not a `date` or `datetime` instance
    '''

    bounds = (train_start, train_end, val_start, val_end, test_start, test_end)
    for name, value in zip(
        ('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end'),
        bounds,
        strict=True,
    ):
        if not isinstance(value, date):
            raise TypeError(
                f"splits {name} must be a date or datetime instance, got {type(value).__name__}: {value!r}"
            )

    if not (train_start < train_end <= val_start < val_end <= test_start < test_end):
        raise ValueError('Date windows must be ordered and non-overlapping')
    return [
        data.filter((pl.col(time_col) >= train_start) & (pl.col(time_col) < train_end)),
        data.filter((pl.col(time_col) >= val_start)   & (pl.col(time_col) < val_end)),
        data.filter((pl.col(time_col) >= test_start)  & (pl.col(time_col) < test_end)),
    ]



def _validate_ratios(ratios):
    if not ratios or any(isinstance(value, bool) or not isinstance(value, numbers.Real)
                         or not math.isfinite(float(value)) or not value >= 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError('Ratios must be non-negative numbers with a positive sum')


def split_data_to_prep_output(split_data, cols=None, all_datetimes=None, *, target_cols=None,
                              time_col=None, as_numpy=False):
    """Create generic train/val/test inputs from caller-owned, already split tables."""
    if len(split_data) != 3:
        raise ValueError('Provide train, validation and test splits')
    columns = list(cols or split_data[0].columns)
    if time_col is not None:
        columns = [column for column in columns if column != time_col]
    if target_cols is None:
        target_cols = columns[-1:]
    single_target = isinstance(target_cols, str) or len(target_cols) == 1
    targets = [target_cols] if isinstance(target_cols, str) else list(target_cols)
    features = [column for column in columns if column not in targets]
    if not features or not targets:
        raise ValueError('Provide feature columns and target columns')
    output = {}
    for name, split in zip(('train', 'val', 'test'), split_data, strict=True):
        x = split.select(features)
        y = split[targets[0]] if single_target else split.select(targets)
        output['x_' + name] = x.to_numpy() if as_numpy else x
        output['y_' + name] = y.to_numpy() if as_numpy else y
    return output
