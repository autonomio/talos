import polars as pl


def mad_transform(df: pl.DataFrame, *, time_col: str = "datetime"):

    '''
    Compute Median Absolute Deviation (MAD) Transform.

    Args:
        df (pl.DataFrame): The input DataFrame.
        time_col (str): The name of the time column.

    Returns:
        pl.DataFrame: The transformed DataFrame.
    '''

    num_cols = [c for c, dt in zip(df.columns, df.dtypes, strict=True)
                if dt.is_numeric() and c != time_col]

    med = df.select([pl.col(c).median().alias(c) for c in num_cols])
    med_vals = {c: float(med[0, c]) for c in num_cols}

    mad = df.select([
        (pl.col(c) - med_vals[c]).abs().median().alias(c) for c in num_cols
    ])
    mad_vals = {c: float(mad[0, c]) or 1.0 for c in num_cols}

    scaled_exprs = [
        ((pl.col(c) - med_vals[c]) / mad_vals[c]).alias(c) for c in num_cols
    ]

    other_exprs = [pl.col(c) for c in df.columns if c not in num_cols]

    return df.select(other_exprs + scaled_exprs)
