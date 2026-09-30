from typing import cast

import polars as pl


class CausalRollingRobustScaler:

    '''Causal trailing-window median and IQR scaling, with no look-ahead.'''

    def __init__(self,
                 x_train: pl.DataFrame,
                 window: int = 1000,
                 quantile_range: tuple[float, float] = (0.25, 0.75),
                 clip: float = 8.0,
                 min_samples: int = 50) -> None:

        '''
        Fit fallback median and IQR statistics on training data.

        The fitted statistics are not applied directly. They are the warmup
        fallback used for rows whose trailing window holds fewer than
        `min_samples` observations, where this scaler degrades gracefully to
        plain median and IQR scaling.

        Args:
            x_train (pl.DataFrame): Training data
            window (int): Trailing window length, in rows, for rolling statistics
            quantile_range (tuple[float, float]): Lower and upper quantiles for IQR
            clip (float): Symmetric bound applied to the scaled output
            min_samples (int): Minimum trailing observations before a rolling statistic is used

        '''

        super().__init__()

        q_low, q_high = quantile_range
        if not (0 <= q_low < q_high <= 1):
            raise ValueError(
                f"CausalRollingRobustScaler quantile_range must satisfy 0 <= low < high <= 1, got ({q_low}, {q_high})"
            )
        if window <= 1:
            raise ValueError(f"CausalRollingRobustScaler window must be >= 2, got {window}")
        if clip <= 0:
            raise ValueError(f"CausalRollingRobustScaler clip must be > 0, got {clip}")
        if not (1 <= min_samples <= window):
            raise ValueError(
                f"CausalRollingRobustScaler min_samples must satisfy 1 <= min_samples <= window ({window}), got {min_samples}"
            )

        self.window = window
        self.quantile_range = quantile_range
        self.clip = clip
        self.min_samples = min_samples

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


    @property
    def context_rows(self) -> int:

        '''Number of raw preceding rows needed to produce a fully warm scaled output.'''

        return self.window


    def transform(self, df: pl.DataFrame) -> pl.DataFrame:

        '''
        Apply causal rolling robust scaling: (x - rolling_median) / rolling_IQR.

        For each row the median and IQR are taken from the trailing `window`
        rows strictly before it (`.shift(1)`), so the transform never reads
        the current row or any future row. Rows with fewer than `min_samples`
        trailing observations fall back to the train-fitted median and IQR.
        The scaled output is clipped to +/- `clip`.

        Args:
            df (pl.DataFrame): Data to transform

        Returns:
            pl.DataFrame: Transformed data
        '''

        q_low, q_high = self.quantile_range
        exprs: list[pl.Expr] = []

        for col in df.columns:
            if col not in self.medians:
                continue

            center = (
                pl.col(col)
                .rolling_median(window_size=self.window, min_samples=self.min_samples)
                .shift(1)
                .fill_null(self.medians[col])
            )
            q_hi = pl.col(col).rolling_quantile(
                q_high, window_size=self.window, min_samples=self.min_samples,
            ).shift(1)
            q_lo = pl.col(col).rolling_quantile(
                q_low, window_size=self.window, min_samples=self.min_samples,
            ).shift(1)

            scale = (q_hi - q_lo).fill_null(self.iqrs[col])
            scale = pl.when(scale > 0).then(scale).otherwise(self.iqrs[col])

            exprs.append(
                ((pl.col(col) - center) / scale).clip(-self.clip, self.clip).alias(col)
            )

        if not exprs:
            return df

        return df.with_columns(exprs)
