"""Map caller observations to Gaussian ranks with reversible fitted quantiles."""

from statistics import NormalDist
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl


class RankGaussScaler:

    '''Rank transformation to Gaussian distribution via inverse normal CDF.'''

    def __init__(self, x_train: pl.DataFrame,
                 n_quantiles: int = 1000) -> None:

        '''
        Fit scaler by storing training quantile boundaries per column.

        Args:
            x_train (pl.DataFrame): Training data
            n_quantiles (int): Number of quantile bins for rank interpolation

        '''

        super().__init__()

        if n_quantiles < 1:
            raise ValueError(
                f"RankGaussScaler n_quantiles must be >= 1, got {n_quantiles}"
            )

        self._n_quantiles = n_quantiles
        self.quantiles: dict[str, npt.NDArray[np.floating[Any]]] = {}

        quantile_points = np.linspace(0, 1, n_quantiles + 1)

        for col in x_train.columns:
            if x_train[col].dtype == pl.Datetime or not x_train[col].dtype.is_numeric():
                continue

            values = x_train[col].cast(pl.Float64).drop_nulls().drop_nans().to_numpy()
            if len(values) == 0:
                continue

            quantiles = np.quantile(values, quantile_points)
            if quantiles[0] == quantiles[-1]:
                continue

            self.quantiles[col] = quantiles

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:

        '''
        Compute rank-based Gaussian transformation.

        For each column: rank values against training quantiles,
        convert to uniform [eps, 1-eps], apply inverse normal CDF.
        NaN values are preserved as NaN in the output.

        Args:
            df (pl.DataFrame): Data to transform

        Returns:
            pl.DataFrame: Transformed data with approximately normal distributions
        '''

        eps = 1.0 / (2 * self._n_quantiles)
        transformed_cols: list[pl.Series] = []

        for col, quantiles in self.quantiles.items():
            if col not in df.columns:
                continue

            values = df[col].cast(pl.Float64).to_numpy()
            nan_mask = np.isnan(values)

            reference = np.linspace(0, 1, len(quantiles))
            uniform = np.clip(np.interp(values, quantiles, reference), eps, 1 - eps)
            gaussian = np.vectorize(lambda probability: NormalDist().inv_cdf(probability)
                                    if np.isfinite(probability) else np.nan, otypes=[float])(uniform)
            gaussian[nan_mask] = np.nan

            transformed_cols.append(pl.Series(col, gaussian))

        if not transformed_cols:
            return df

        return df.with_columns(transformed_cols)


def inverse_transform(df: pl.DataFrame, scaler: RankGaussScaler) -> pl.DataFrame:

    '''
    Reverse the rank-based Gaussian transformation.

    Applies normal CDF to convert Gaussian values back to uniform,
    then interpolates against stored training quantiles to recover
    approximate original values. The inverse is approximate because
    rank-based transforms are lossy.

    Args:
        df (pl.DataFrame): Gaussian-transformed data
        scaler (RankGaussScaler): Fitted scaler instance with quantiles

    Returns:
        pl.DataFrame: Data in approximate original scale
    '''

    restored_cols: list[pl.Series] = []

    for col, quantiles in scaler.quantiles.items():
        if col not in df.columns:
            continue

        gaussian = df[col].to_numpy().astype(np.float64)
        nan_mask = np.isnan(gaussian)

        uniform = np.vectorize(NormalDist().cdf, otypes=[float])(gaussian)
        reference = np.linspace(0, 1, len(quantiles))
        original = np.interp(uniform, reference, quantiles)
        original[nan_mask] = np.nan

        restored_cols.append(pl.Series(col, original))

    if not restored_cols:
        return df

    return df.with_columns(restored_cols)
