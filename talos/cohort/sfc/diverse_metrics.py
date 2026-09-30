from typing import Any
from typing import Protocol
from typing import TypeGuard

import numpy as np
import numpy.typing as npt
import polars as pl


class _FittedScaler(Protocol):

    '''Typed facade over the sklearn StandardScaler surface used for selection.'''

    def fit_transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.floating[Any]]: ...


class _Projector(Protocol):

    '''Typed facade over the sklearn PCA surface used for selection.'''

    def fit_transform(self, X: npt.NDArray[Any]) -> npt.NDArray[np.floating[Any]]: ...


class _Clusterer(Protocol):

    '''Typed facade over the sklearn KMeans surface used for selection.'''

    @property
    def cluster_centers_(self) -> npt.NDArray[np.floating[Any]]: ...

    def fit_predict(self, X: npt.NDArray[Any]) -> npt.NDArray[np.integer[Any]]: ...


def _standard_scaler() -> _FittedScaler:

    '''Create a sklearn StandardScaler behind the typed facade.'''

    from sklearn.preprocessing import StandardScaler
    return StandardScaler()


def _pca(n_components: int | None, random_state: int) -> _Projector:

    '''Create a sklearn PCA projector behind the typed facade.'''

    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=random_state)


def _kmeans(n_clusters: int, random_state: int) -> _Clusterer:

    '''Create a sklearn KMeans clusterer behind the typed facade.'''

    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')


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
           iqr_multiplier: float = 3.0,
           n_components: object = None,
           n_clusters: object = 8,
           random_state: int = 42) -> list[int | str]:
    '''
    Return a metric-diverse set of ids using PCA/KMeans medoid selection.

    Metric columns are caller supplied or numeric columns from results.
    '''

    if not isinstance(target_count, int) or target_count <= 0:
        raise ValueError('diverse_metrics target_count must be a positive integer')
    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError('diverse_metrics n_clusters must be a positive integer')
    if n_components is not None and (
        not isinstance(n_components, int) or n_components <= 0
    ):
        raise ValueError('diverse_metrics n_components must be a positive integer')
    if iqr_multiplier < 0:
        raise ValueError('diverse_metrics iqr_multiplier must be >= 0')

    min_default_metric_cols = 2
    results = context.get('results')
    if results is None:
        raise ValueError('diverse_metrics selector requires results.csv data in context["results"]')
    if not isinstance(results, pl.DataFrame):
        raise ValueError('diverse_metrics context["results"] must be a polars DataFrame')

    if metric_cols is None:
        metric_cols = [name for name, dtype in results.schema.items()
                       if name != 'id' and dtype.is_numeric()]
        if len(metric_cols) < min_default_metric_cols:
            raise ValueError('diverse_metrics requires at least two numeric metric columns')

    missing = [col for col in ['id', *metric_cols] if col not in results.columns]
    if missing:
        raise ValueError(f'diverse_metrics selector input is missing required columns: {missing}')

    def coerce_id(value: Any) -> int | str:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('diverse_metrics selector returned a boolean permutation id')
        if value is None or (_is_floating(value) and np.isnan(float(value))):
            raise ValueError('diverse_metrics selector returned a missing permutation id')
        if _is_integral(value):
            return int(value)
        if _is_floating(value) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
            if not stripped:
                raise ValueError('diverse_metrics selector returned an empty permutation id')
            return stripped

        coerced = str(value).strip()
        if not coerced:
            raise ValueError('diverse_metrics selector returned an empty permutation id')
        return coerced

    work = results.select(['id', *metric_cols]).with_columns(
        [pl.col(col).cast(pl.Float64, strict=False) for col in metric_cols]
    )
    work = work.filter(
        pl.all_horizontal([pl.col(col).is_not_null() & pl.col(col).is_not_nan() for col in metric_cols])
    )
    if work.is_empty():
        return []

    filtered = work
    for col in metric_cols:
        col_values = filtered[col].to_numpy()
        if col_values.size == 0:
            continue
        q1 = np.quantile(col_values, 0.25)
        q3 = np.quantile(col_values, 0.75)
        if q1 == q3:
            continue
        iqr = q3 - q1
        lo = float(q1 - iqr_multiplier * iqr)
        hi = float(q3 + iqr_multiplier * iqr)
        filtered = filtered.filter((pl.col(col) >= lo) & (pl.col(col) <= hi))
    if not filtered.is_empty():
        work = filtered

    if len(work) <= target_count:
        return [coerce_id(value) for value in work['id'].to_list()]

    values = work.select(metric_cols).to_numpy()
    n_samples, n_features = values.shape
    actual_clusters = min(n_clusters, n_samples)
    actual_components = (
        None
        if n_components is None
        else min(n_components, n_samples, n_features)
    )

    scaled = _standard_scaler().fit_transform(values)
    projected = _pca(actual_components, random_state).fit_transform(scaled)

    if actual_clusters == 1:
        center = projected.mean(axis=0)
        distances = np.linalg.norm(projected - center, axis=1)
        selected_positions = [int(np.argmin(distances))]
    else:
        kmeans = _kmeans(actual_clusters, random_state)
        labels = kmeans.fit_predict(projected)

        selected_positions: list[int] = []
        for cluster_id in range(actual_clusters):
            mask = labels == cluster_id
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue
            cluster_points = projected[mask]
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            selected_positions.append(int(idxs[int(np.argmin(distances))]))

    if len(selected_positions) < target_count:
        parts: list[npt.NDArray[np.float64]] = []
        for col in metric_cols:
            col_values = work[col].to_numpy()
            lo = np.nanmin(col_values)
            hi = np.nanmax(col_values)
            if not np.isfinite(lo) or not np.isfinite(hi):
                parts.append(np.zeros(len(col_values), dtype=float))
            elif hi == lo:
                parts.append(np.ones(len(col_values), dtype=float))
            else:
                parts.append((col_values - lo) / (hi - lo))

        scores = np.vstack(parts).mean(axis=0)
        for ranked_idx in np.argsort(-scores):
            if len(selected_positions) >= target_count:
                break
            idx = int(ranked_idx)
            if idx not in selected_positions:
                selected_positions.append(idx)

    selected_positions = list(dict.fromkeys(selected_positions))[:target_count]
    return [
        coerce_id(value)
        for value in work['id'].gather(selected_positions).to_list()
    ]
