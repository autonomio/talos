import polars as pl


def shift_column_transform(data: pl.DataFrame, shift: int, column: str) -> pl.DataFrame:

    '''
    Shift a column by a specified number of periods.

    Args:
        data (pl.DataFrame): Input DataFrame
        shift (int): Number of periods to shift (negative for forward shift)
        column (str): Name of column to shift

    Returns:
        pl.DataFrame: DataFrame with shifted column
    '''

    return data.with_columns(
        pl.col(column).shift(shift).alias(column)
    )
