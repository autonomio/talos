import logging
from typing import Any, Literal

import polars as pl
from typing_extensions import override

from talos.experiment.reducer.pruning_strategy import ACTION_SUGGEST
from talos.experiment.reducer.pruning_strategy import PruningStrategy
from talos.experiment.reducer.correlation_analysis import experiment_parameter_correlation

_CorrelationMethod = Literal['pearson', 'kendall', 'spearman']

logger = logging.getLogger(__name__)


class _CorrelationShim:

    '''Minimal wrapper to satisfy experiment_parameter_correlation's self.experiment_log interface.'''

    def __init__(self, experiment_log: Any) -> None:

        super().__init__()

        self.experiment_log = experiment_log


class CorrelationReducer(PruningStrategy):

    '''
    Prune parameter values based on bootstrap correlation analysis.

    Two pruning modes:
    - Wrong-direction: parameters with strong negative correlation
      (for maximization) get their worst-performing value removed
    - Low-impact: parameters with negligible |correlation| and high
      sign stability get a keep_is suggestion for the best value

    NOTE: Only acts on numeric parameters. Categorical/string parameters
    are coerced to numeric internally and silently skipped when coercion
    produces all-NaN or constant columns.

    '''

    def __init__(self,
                 *,
                 metric: str,
                 method: str = 'spearman',
                 min_observations: int = 50,
                 prune_threshold: float = 0.05,
                 sign_stability_threshold: float = 0.8,
                 n_boot: int = 300,
                 negative_correlation_threshold: float = -0.3,
                 maximize: bool = True,
                 random_state: int = 0,
                 active: bool = True) -> None:

        '''
        Initialize the CorrelationReducer.

        Args:
            metric (str): Target metric column for correlation analysis
            method (str): Correlation method ('spearman', 'pearson', 'kendall')
            min_observations (int): Minimum rows before correlation is computed
            prune_threshold (float): |correlation| below this is considered
                low-impact (eligible for keep_is suggestion)
            sign_stability_threshold (float): Minimum sign stability to act
            n_boot (int): Number of bootstrap resamples
            negative_correlation_threshold (float): Correlation below this
                triggers wrong-direction removal (for maximize=True)
            maximize (bool): Whether higher metric values are better
            random_state (int): RNG seed for bootstrap reproducibility
            active (bool): Whether this reducer is enabled

        '''

        valid_methods = ('spearman', 'pearson', 'kendall')
        if method not in valid_methods:
            raise ValueError(
                f"CorrelationReducer method must be one of {valid_methods}, got '{method}'"
            )

        if min_observations <= 0:
            raise ValueError(
                f"CorrelationReducer min_observations must be > 0, got {min_observations}"
            )

        if not 0.0 <= prune_threshold <= 1.0:
            raise ValueError(
                f"CorrelationReducer prune_threshold must be between 0.0 and 1.0, got {prune_threshold}"
            )

        if not 0.0 <= sign_stability_threshold <= 1.0:
            raise ValueError(
                f"CorrelationReducer sign_stability_threshold must be between 0.0 and 1.0, got {sign_stability_threshold}"
            )

        if n_boot <= 0:
            raise ValueError(
                f"CorrelationReducer n_boot must be > 0, got {n_boot}"
            )

        if not -1.0 <= negative_correlation_threshold <= 0.0:
            raise ValueError(
                f"CorrelationReducer negative_correlation_threshold must be between -1.0 and 0.0, got {negative_correlation_threshold}"
            )

        super().__init__(active=active)
        self._metric = metric
        self._method: _CorrelationMethod = method
        self._min_observations = min_observations
        self._prune_threshold = prune_threshold
        self._sign_stability_threshold = sign_stability_threshold
        self._n_boot = n_boot
        self._negative_correlation_threshold = negative_correlation_threshold
        self._maximize = maximize
        self._random_state = random_state
        self._applied: set[tuple[str, Any]] = set()
        self._suggested: set[str] = set()


    @override
    def analyze_and_intervene(self,
                              log: pl.DataFrame,
                              msq: Any) -> list[dict[str, Any]]:

        '''
        Analyze parameter correlations and return interventions.

        Computes bootstrap correlations between each parameter and the
        target metric. Emits remove_is for wrong-direction parameters
        and keep_is suggestions for low-impact parameters.

        '''

        if not self._active:
            return []

        df = log

        if df.is_empty() or self._metric not in df.columns:
            return []

        if len(df) < self._min_observations:
            return []

        try:
            corr_df = self._compute_correlations(df, msq.domain_keys)
        except (ValueError, KeyError):
            logger.debug('Correlation computation failed, skipping')
            return []

        if corr_df.is_empty():
            return []

        domain_params = set(msq.domain_keys)
        interventions: list[dict[str, Any]] = []
        filtered = self._filter_valid_metric(df)

        for row in corr_df.iter_rows(named=True):
            param = row['feature']

            if param not in domain_params:
                continue

            corr_med = row['corr_med']
            sign_stability = row['sign_stability']

            if sign_stability < self._sign_stability_threshold:
                continue

            value_means = self._value_means(filtered, param)
            if not value_means:
                continue

            interventions.extend(
                self._check_wrong_direction(param, corr_med, value_means)
            )
            interventions.extend(
                self._check_low_impact(param, corr_med, value_means)
            )

        return interventions


    def _compute_correlations(self,
                              df: pl.DataFrame,
                              domain_keys: list[str]) -> pl.DataFrame:

        '''
        Compute bootstrap correlations via the existing correlation function.

        Args:
            df (pl.DataFrame): Experiment log
            domain_keys (list[str]): Parameter names from the domain

        Returns:
            pl.DataFrame: Correlation results with columns 'feature',
                'corr_med', 'sign_stability'

        '''

        cols = [c for c in domain_keys if c in df.columns] + [self._metric]
        import pandas as pd
        pdf = pd.DataFrame(df.select(cols).to_dict(as_series=False))
        shim = _CorrelationShim(pdf)

        result = experiment_parameter_correlation(
            shim,
            self._metric,
            heads=(1.0,),
            method=self._method,
            n_boot=self._n_boot,
            min_n=self._min_observations,
            random_state=self._random_state,
        )

        return pl.DataFrame(result.reset_index().to_dict(orient='list'))


    def _check_wrong_direction(self,
                               param: str,
                               corr_med: float,
                               value_means: dict[Any, float]) -> list[dict[str, Any]]:

        '''Emit remove_is for the worst-performing value if correlation is wrong-direction.'''

        if self._maximize:
            is_wrong = corr_med < self._negative_correlation_threshold
        else:
            is_wrong = corr_med > abs(self._negative_correlation_threshold)

        if not is_wrong:
            return []

        if self._maximize:
            worst_value = min(value_means, key=lambda v: value_means[v])
        else:
            worst_value = max(value_means, key=lambda v: value_means[v])

        if (param, worst_value) in self._applied:
            return []

        self._applied.add((param, worst_value))
        direction = 'negative' if self._maximize else 'positive'
        return [{
            'op': 'remove_is',
            'param': param,
            'value': worst_value,
            'reason': f"wrong-direction ({direction}) correlation {corr_med:.3f} for {param}",
        }]


    def _check_low_impact(self,
                          param: str,
                          corr_med: float,
                          value_means: dict[Any, float]) -> list[dict[str, Any]]:

        '''Emit keep_is suggestion for the best value if parameter has negligible impact.'''

        if abs(corr_med) >= self._prune_threshold:
            return []

        if param in self._suggested:
            return []

        if self._maximize:
            best_value = max(value_means, key=lambda v: value_means[v])
        else:
            best_value = min(value_means, key=lambda v: value_means[v])

        self._suggested.add(param)
        return [{
            'op': 'keep_is',
            'param': param,
            'value': best_value,
            'action': ACTION_SUGGEST,
            'reason': f"low-impact correlation {corr_med:.3f} for {param}",
        }]


    def _filter_valid_metric(self, df: pl.DataFrame) -> pl.DataFrame:

        '''Filter rows with valid (non-null, non-NaN) metric values.'''

        col = pl.col(self._metric)
        filter_expr = col.is_not_null()
        if df[self._metric].dtype.is_float():
            filter_expr = filter_expr & col.is_not_nan()
        return df.filter(filter_expr)


    def _value_means(self,
                     filtered: pl.DataFrame,
                     param: str) -> dict[Any, float]:

        '''Compute mean metric per parameter value from pre-filtered data.'''

        if param not in filtered.columns:
            return {}

        stats = (
            filtered
              .group_by(param)
              .agg(pl.col(self._metric).mean().alias('_mean'))
        )

        return {
            row[param]: row['_mean']
            for row in stats.iter_rows(named=True)
        }


    @override
    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        return {
            'applied': list(self._applied),
            'suggested': list(self._suggested),
        }


    @override
    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        required = ('applied', 'suggested')
        for key in required:
            if key not in state:
                raise ValueError(
                    f"Invalid CorrelationReducer state: missing required key '{key}'."
                )

        self._applied = set(tuple(x) for x in state['applied'])
        self._suggested = set(state['suggested'])
