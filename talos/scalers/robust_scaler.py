from typing import cast

import polars as pl


class RobustScaler:

    '''Median and IQR scaling, resilient to outliers.'''

    def __init__(self,
                 x_train: pl.DataFrame,
                 quantile_range: tuple[float, float] = (0.25, 0.75)) -> None:

        '''
        Fit scaler on training data using median and interquartile range.

        Args:
            x_train (pl.DataFrame): Training data
            quantile_range (tuple[float, float]): Lower and upper quantiles for IQR

        '''

        super().__init__()

        q_low, q_high = quantile_range
        if not (0 <= q_low < q_high <= 1):
            raise ValueError(
                f"RobustScaler quantile_range must satisfy 0 <= low < high <= 1, got ({q_low}, {q_high})"
            )

        self.medians: dict[str, float] = {}
        self.iqrs: dict[str, float] = {}

        for col in x_train.columns:
            if x_train[col].dtype == pl.Datetime or not x_train[col].dtype.is_numeric():
                continue

            median = x_train[col].median()
            q_lo = x_train[col].quantile(q_low)
            q_hi = x_train[col].quantile(q_high)

            if median is None or q_lo is None or q_hi is None:
                continue

            iqr = q_hi - q_lo

            self.medians[col] = cast(float, median)
            self.iqrs[col] = iqr if iqr != 0 else 1.0


    def transform(self, df: pl.DataFrame) -> pl.DataFrame:

        '''
        Apply robust scaling: (x - median) / IQR.

        Args:
            df (pl.DataFrame): Data to transform

        Returns:
            pl.DataFrame: Transformed data
        '''

        exprs: list[pl.Expr] = []
        for col in df.columns:
            if col in self.medians:
                exprs.append(
                    ((pl.col(col) - self.medians[col]) / self.iqrs[col]).alias(col)
                )

        if not exprs:
            return df

        return df.with_columns(exprs)


def inverse_transform(df: pl.DataFrame, scaler: RobustScaler) -> pl.DataFrame:

    '''
    Reverse the robust scaling transformation.

    Args:
        df (pl.DataFrame): Scaled data to inverse transform
        scaler (RobustScaler): Fitted scaler instance with parameters

    Returns:
        pl.DataFrame: Data in original scale
    '''

    exprs: list[pl.Expr] = []
    for col in df.columns:
        if col in scaler.medians:
            exprs.append(
                (pl.col(col) * scaler.iqrs[col] + scaler.medians[col]).alias(col)
            )

    if not exprs:
        return df

    return df.with_columns(exprs)
