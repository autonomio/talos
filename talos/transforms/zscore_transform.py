import polars as pl


def zscore_transform(df: pl.DataFrame, *, time_col: str = 'datetime') -> pl.DataFrame:

    '''
    Compute standard Z-score scaling for numeric columns.

    Args:
        df (pl.DataFrame): Klines dataset with numeric columns to scale
        time_col (str): Column name to exclude from numeric transforms

    Returns:
        pl.DataFrame: The input data with Z-scored numeric columns
    '''

    num_cols = [c for c, dt in zip(df.columns, df.dtypes, strict=True)
                if dt.is_numeric() and c != time_col]

    if not num_cols:
        return df

    stats = df.select([
        pl.col(c).mean().alias(f'{c}__mu') for c in num_cols
    ] + [
        pl.col(c).std().alias(f'{c}__sd') for c in num_cols
    ])

    mu = {c: float(stats[0, f'{c}__mu']) for c in num_cols}
    sd = {c: float(stats[0, f'{c}__sd']) or 1.0 for c in num_cols}

    scaled_exprs = [
        ((pl.col(c) - mu[c]) / sd[c]).alias(c) for c in num_cols
    ]

    other_exprs = [pl.col(c) for c in df.columns if c not in num_cols]

    return df.select(other_exprs + scaled_exprs)
