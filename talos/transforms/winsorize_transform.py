import polars as pl


def winsorize_transform(df: pl.DataFrame, *, time_col: str = 'datetime') -> pl.DataFrame:

    '''
    Compute winsorization by clipping numeric columns to fixed quantile bounds.

    Args:
        df (pl.DataFrame): Klines dataset with numeric columns to clip
        time_col (str): Column name to exclude from numeric transforms

    Returns:
        pl.DataFrame: The input data with winsorized numeric columns
    '''

    # Numeric columns excluding the time column
    num_cols = [c for c, dt in zip(df.columns, df.dtypes, strict=True)
                if dt.is_numeric() and c != time_col]

    if not num_cols:
        return df

    # Fixed quantile bounds for winsorization
    lower_q = 0.01
    upper_q = 0.99

    # Compute per-column quantiles
    lower_sel = df.select([pl.col(c).quantile(lower_q).alias(c) for c in num_cols])
    upper_sel = df.select([pl.col(c).quantile(upper_q).alias(c) for c in num_cols])
    lower = {c: float(lower_sel[0, c]) for c in num_cols}
    upper = {c: float(upper_sel[0, c]) for c in num_cols}

    # Build clipping expressions
    clipped_exprs = [
        pl.col(c).clip(lower[c], upper[c]).alias(c) for c in num_cols
    ]

    # Preserve non-numeric columns as-is
    other_exprs = [pl.col(c) for c in df.columns if c not in num_cols]

    return df.select(other_exprs + clipped_exprs)
