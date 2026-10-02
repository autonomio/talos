"""Select bounded contiguous slices from caller DataFrames."""

import numpy as np
import polars as pl


def random_slice(df: pl.DataFrame,
                 rows: int, *,
                 safe_range_low: float = 0.25,
                 safe_range_high: float = 0.75,
                 seed: int | None = None) -> pl.DataFrame:

    '''
    Compute contiguous slice from DataFrame within specified safe range.

    Args:
        df (pl.DataFrame): Input DataFrame to slice from
        rows (int): Number of rows to include in the slice
        safe_range_low (float): Lower bound of safe range as fraction of total rows
        safe_range_high (float): Upper bound of safe range as fraction of total rows
        seed (int | None): Random seed for reproducible results

    Returns:
        pl.DataFrame: Contiguous slice of the original DataFrame maintaining row order
    '''
    # Validate safe range parameters
    if not (0.0 <= safe_range_low < safe_range_high <= 1.0):
        raise ValueError('random_slice safe_range_low must be >= 0.0, safe_range_high must be <= 1.0, and safe_range_low < safe_range_high')

    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError('random_slice rows must be a positive integer')
    n = len(df)
    lo = int(n * safe_range_low)
    hi = int(n * safe_range_high) - rows  # highest valid start

    if hi < lo:
        raise ValueError(f'random_slice slice size ({rows}) too large for chosen safe range ({safe_range_low * 100:.0f}%-{safe_range_high * 100:.0f}%)')

    rng = np.random.default_rng(seed)
    start = int(rng.integers(lo, hi + 1))
    return df[start : start + rows]  # slice keeps order
