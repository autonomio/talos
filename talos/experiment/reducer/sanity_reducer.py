from typing import Any

import polars as pl
from typing_extensions import override

from talos.experiment.reducer.pruning_strategy import ACTION_SUGGEST
from talos.experiment.reducer.pruning_strategy import PruningStrategy


class SanityReducer(PruningStrategy):

    '''
    Remove parameter values whose trials produce null or NaN in the
    target metric above a configurable threshold.

    Counts both null and NaN (for float columns) in the metric column.
    Values are pruned when missing_count / total > nan_threshold and
    total >= min_observations.

    Optionally emits suggestion interventions (logged but not dispatched)
    for zero-metric, execution timeout, and warning rate signals.

    '''

    def __init__(self,
                 *,
                 metric: str,
                 nan_threshold: float = 0.1,
                 min_observations: int = 1,
                 zero_metric_threshold: float | None = None,
                 execution_time_column: str = 'execution_time',
                 execution_time_threshold: float | None = None,
                 timeout_rate_threshold: float = 0.5,
                 warning_threshold: float | None = None,
                 active: bool = True) -> None:

        '''
        Initialize the SanityReducer.

        Args:
            metric (str): Name of the target metric column to check for NaN
            nan_threshold (float): NaN rate above which a value is pruned
                (strict >)
            min_observations (int): Minimum trial count before a value can
                be considered for pruning
            zero_metric_threshold (float | None): Suggest if zero-metric
                rate exceeds this (None to disable)
            execution_time_column (str): Column name for execution time
            execution_time_threshold (float | None): Suggest if execution
                time exceeds this value (None to disable)
            timeout_rate_threshold (float): Fraction of trials exceeding
                execution_time_threshold to trigger suggestion
            warning_threshold (float | None): Suggest if warning rate
                exceeds this (None to disable)
            active (bool): Whether this reducer is enabled

        '''

        if not 0.0 <= nan_threshold <= 1.0:
            raise ValueError(
                f"SanityReducer nan_threshold must be between 0.0 and 1.0, got {nan_threshold}"
            )

        if min_observations <= 0:
            raise ValueError(
                f"SanityReducer min_observations must be > 0, got {min_observations}"
            )

        if zero_metric_threshold is not None and not 0.0 <= zero_metric_threshold <= 1.0:
            raise ValueError(
                f"SanityReducer zero_metric_threshold must be between 0.0 and 1.0, got {zero_metric_threshold}"
            )

        if not 0.0 <= timeout_rate_threshold <= 1.0:
            raise ValueError(
                f"SanityReducer timeout_rate_threshold must be between 0.0 and 1.0, got {timeout_rate_threshold}"
            )

        if warning_threshold is not None and not 0.0 <= warning_threshold <= 1.0:
            raise ValueError(
                f"SanityReducer warning_threshold must be between 0.0 and 1.0, got {warning_threshold}"
            )

        super().__init__(active=active)
        self._metric = metric
        self._nan_threshold = nan_threshold
        self._min_observations = min_observations
        self._zero_metric_threshold = zero_metric_threshold
        self._execution_time_column = execution_time_column
        self._execution_time_threshold = execution_time_threshold
        self._timeout_rate_threshold = timeout_rate_threshold
        self._warning_threshold = warning_threshold
        self._removed: set[tuple[str, Any]] = set()
        self._suggested: set[tuple[str, Any, str]] = set()


    @override
    def analyze_and_intervene(self,
                              log: pl.DataFrame,
                              msq: Any) -> list[dict[str, Any]]:

        '''
        Analyze experiment log and return interventions.

        Returns remove_is interventions for NaN rate exceeding the
        threshold, plus suggestion interventions for zero-metric,
        execution timeout, and warning rate signals.

        '''

        if not self._active:
            return []

        df = log

        if df.is_empty() or self._metric not in df.columns:
            return []

        param_names = msq.domain_keys
        interventions: list[dict[str, Any]] = []

        is_float = df[self._metric].dtype.is_float()
        nan_expr = pl.col(self._metric).is_null()
        if is_float:
            nan_expr = nan_expr | pl.col(self._metric).is_nan()

        base_exprs: list[pl.Expr] = [
            pl.len().alias('_total'),
            nan_expr.sum().alias('_nan_count'),
        ]
        detectors: list[tuple[str, float, str]] = []

        if self._zero_metric_threshold is not None:
            base_exprs.append(
                (pl.col(self._metric) == 0.0).sum().alias('_zero_count')
            )
            detectors.append(('_zero_count', self._zero_metric_threshold, 'zero_metric'))

        if (self._execution_time_threshold is not None
                and self._execution_time_column in df.columns):
            base_exprs.append(
                (pl.col(self._execution_time_column) > self._execution_time_threshold)
                    .sum().alias('_timeout_count')
            )
            detectors.append(('_timeout_count', self._timeout_rate_threshold, 'execution_timeout'))

        if (self._warning_threshold is not None
                and '_warnings' in df.columns):
            base_exprs.append(
                (pl.col('_warnings') != '[]').sum().alias('_warning_count')
            )
            detectors.append(('_warning_count', self._warning_threshold, 'warning'))

        for param in param_names:
            if param not in df.columns:
                continue

            stats = df.group_by(param).agg(*base_exprs)

            for row in stats.iter_rows(named=True):
                value = row[param]

                if row['_total'] < self._min_observations:
                    continue

                if ((param, value) not in self._removed
                        and row['_nan_count'] / row['_total'] > self._nan_threshold):
                    self._removed.add((param, value))
                    interventions.append({
                        'op': 'remove_is',
                        'param': param,
                        'value': value,
                    })

                for count_col, threshold, label in detectors:
                    interventions.extend(
                        self._rate_suggestion(row, param, value, count_col, threshold, label)
                    )

        return interventions


    def _rate_suggestion(self,
                         row: dict[str, Any],
                         param: str,
                         value: Any,
                         count_col: str,
                         threshold: float,
                         reason_label: str) -> list[dict[str, Any]]:

        '''Emit a suggestion if the rate exceeds the threshold.'''

        if (param, value) in self._removed:
            return []

        if (param, value, reason_label) in self._suggested:
            return []

        rate = row[count_col] / row['_total']
        if rate > threshold:
            self._suggested.add((param, value, reason_label))
            return [{
                'op': 'remove_is',
                'param': param,
                'value': value,
                'action': ACTION_SUGGEST,
                'reason': f"{reason_label} rate {rate:.2f} for {param}={value}",
            }]

        return []


    @override
    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        return {
            'removed': list(self._removed),
            'suggested': list(self._suggested),
        }


    @override
    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        if 'removed' not in state:
            raise ValueError("Invalid SanityReducer state: missing required key 'removed'.")

        self._removed = set(tuple(x) for x in state['removed'])
        self._suggested = set(tuple(x) for x in state.get('suggested', []))
