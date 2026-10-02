"""Focus parameter exploration around observed metric breakthroughs."""

import logging
from math import ceil, floor
from numbers import Integral, Real
from typing import Any

import polars as pl
from typing_extensions import override

from talos.experiment.reducer.filter_types import FILTER_KEEP_BETWEEN, FILTER_KEEP_VALUES
from talos.experiment.reducer.pruning_strategy import PruningStrategy

logger = logging.getLogger(__name__)


class FocusReducer(PruningStrategy):

    '''Detect breakthrough results and narrow the parameter space for exploitation using reversible named filters.

    Monitors a target metric for values crossing a configurable
    threshold. On breakthrough, narrows each parameter around the
    winning combination and injects fine-grained linspace variations
    for numeric parameters. Snaps back to full exploration after
    focus_timeout rounds without improvement.

    NOTE: Configuring multiple FocusReducer instances that operate on
    overlapping parameters may cause conflicting filters. Use one
    FocusReducer per experiment, or ensure parameter spaces are disjoint.
    '''

    def __init__(self,
                 *,
                 metric: str,
                 breakthrough_threshold: float,
                 maximize: bool = True,
                 focus_range_pct: float = 0.2,
                 focus_timeout: int = 5,
                 variation_count: int = 5,
                 min_observations: int = 10,
                 active: bool = True) -> None:

        '''
        Initialize the FocusReducer.

        Args:
            metric (str): Target metric column to monitor for breakthroughs
            breakthrough_threshold (float): Value that triggers focus mode
            maximize (bool): If True, breakthrough is metric > threshold;
                if False, metric < threshold
            focus_range_pct (float): Half-width as fraction of center value
                for numeric parameter narrowing
            focus_timeout (int): Rounds without improvement before snap-back
            variation_count (int): Number of linspace points to inject per
                numeric parameter on activation
            min_observations (int): Minimum rows in log before analysis
            active (bool): Whether this reducer is enabled

        '''

        if focus_range_pct < 0:
            raise ValueError(
                f"FocusReducer focus_range_pct must be >= 0, got {focus_range_pct}"
            )

        if focus_timeout <= 0:
            raise ValueError(
                f"FocusReducer focus_timeout must be > 0, got {focus_timeout}"
            )

        if variation_count <= 0:
            raise ValueError(
                f"FocusReducer variation_count must be > 0, got {variation_count}"
            )

        if min_observations <= 0:
            raise ValueError(
                f"FocusReducer min_observations must be > 0, got {min_observations}"
            )

        super().__init__(active=active)
        self._metric = metric
        self._breakthrough_threshold = breakthrough_threshold
        self._maximize = maximize
        self._focus_range_pct = focus_range_pct
        self._focus_timeout = focus_timeout
        self._variation_count = variation_count
        self._min_observations = min_observations

        self._focus_active: bool = False
        self._breakthrough_combo: dict[str, Any] | None = None
        self._best_metric: float | None = None
        self._rounds_since_improvement: int = 0
        self._focused_params: set[str] = set()

    @override
    def analyze_and_intervene(self,
                              log: pl.DataFrame,
                              msq: Any) -> list[dict[str, Any]]:

        '''
        Analyze log for breakthrough results and return interventions.

        When OFF, scans for metric crossing the threshold. When ON,
        checks for improvement or timeout snap-back.

        '''

        if not self._active:
            return []

        df = log

        if df.is_empty() or self._metric not in df.columns:
            return []

        if len(df) < self._min_observations:
            return []

        best = self._find_best_row(df, msq.domain_keys)
        if best is None:
            return []

        best_value, best_combo = best
        best_combo = {param: msq.resolve_log_value(param, value)
                      for param, value in best_combo.items()}

        if not self._focus_active:
            return self._handle_off(best_value, best_combo, msq)

        return self._handle_on(best_value, best_combo, msq)

    def _handle_off(self,
                    best_value: float,
                    best_combo: dict[str, Any],
                    msq: Any) -> list[dict[str, Any]]:

        '''Check for breakthrough and activate focus if threshold crossed.'''

        if self._maximize and best_value <= self._breakthrough_threshold:
            return []
        if not self._maximize and best_value >= self._breakthrough_threshold:
            return []

        self._focus_active = True
        self._breakthrough_combo = dict(best_combo)
        self._best_metric = best_value
        self._rounds_since_improvement = 0

        interventions = self._build_narrowing(best_combo, msq)
        interventions.extend(self._build_injections(best_combo, msq))

        return interventions

    def _handle_on(self,
                   best_value: float,
                   best_combo: dict[str, Any],
                   msq: Any) -> list[dict[str, Any]]:

        '''Check for improvement or timeout while focus is active.'''

        if self._best_metric is None:
            raise ValueError('FocusReducer focus is active but no best metric is recorded')

        improved = (
            (self._maximize and best_value > self._best_metric)
            or (not self._maximize and best_value < self._best_metric)
        )

        if improved:
            self._best_metric = best_value
            self._breakthrough_combo = dict(best_combo)
            self._rounds_since_improvement = 0
            return self._build_narrowing(best_combo, msq)

        self._rounds_since_improvement += 1

        if self._rounds_since_improvement >= self._focus_timeout:
            return self._snap_back()

        return []

    def _snap_back(self) -> list[dict[str, Any]]:

        '''Revert to full exploration by clearing all focus filters.'''

        interventions: list[dict[str, Any]] = []
        for key in sorted(self._focused_params):
            interventions.append({
                'op': 'clear_filter',
                'key': key,
                'reason': f"focus timeout after {self._focus_timeout} rounds without improvement",
            })

        self._focus_active = False
        self._breakthrough_combo = None
        self._best_metric = None
        self._rounds_since_improvement = 0
        self._focused_params = set()

        return interventions

    def _find_best_row(self,
                       df: pl.DataFrame,
                       domain_keys: list[str]) -> tuple[float, dict[str, Any]] | None:

        '''
        Find the row with the best metric value.

        Args:
            df (pl.DataFrame): Experiment log
            domain_keys (list[str]): Parameter column names

        Returns:
            tuple[float, dict[str, Any]] | None: Best metric value and
                combo dict, or None if no valid rows

        '''

        col = pl.col(self._metric)
        filter_expr = col.is_not_null()
        if df[self._metric].dtype.is_float():
            filter_expr = filter_expr & col.is_not_nan()
        filtered = df.filter(filter_expr)

        if filtered.is_empty():
            return None

        if self._maximize:
            idx = filtered[self._metric].arg_max()
        else:
            idx = filtered[self._metric].arg_min()

        row = filtered.row(idx, named=True)
        metric_value = row[self._metric]
        combo = {k: row[k] for k in domain_keys if k in row}

        return metric_value, combo

    def _build_narrowing(self,
                         combo: dict[str, Any],
                         msq: Any) -> list[dict[str, Any]]:

        '''
        Emit set_filter interventions to narrow around the combo.

        Numeric params get FILTER_KEEP_BETWEEN, categorical params
        get FILTER_KEEP_VALUES.

        '''

        interventions: list[dict[str, Any]] = []

        for param in msq.domain_keys:
            if param not in combo:
                continue

            value = combo[param]
            key = f"focus_{self._metric}_{param}"
            self._focused_params.add(key)

            if self._is_numeric(value):
                lower, upper = self._focus_bounds(value)
                interventions.append({
                    'op': 'set_filter',
                    'key': key,
                    'filter_type': FILTER_KEEP_BETWEEN,
                    'filter_params': {
                        'param': param,
                        'lower': lower,
                        'upper': upper,
                    },
                    'reason': f"focus narrowing {param} to [{lower}, {upper}] around {value}",
                })
            else:
                interventions.append({
                    'op': 'set_filter',
                    'key': key,
                    'filter_type': FILTER_KEEP_VALUES,
                    'filter_params': {
                        'param': param,
                        'values': [value],
                    },
                    'reason': f"focus narrowing {param} to {value}",
                })

        return interventions

    def _build_injections(self,
                          combo: dict[str, Any],
                          msq: Any) -> list[dict[str, Any]]:

        '''Emit inject_value interventions for numeric params.'''

        interventions: list[dict[str, Any]] = []

        for param in msq.domain_keys:
            if param not in combo:
                continue

            value = combo[param]
            if not self._is_numeric(value):
                continue

            lower, upper = self._focus_bounds(value)

            if self._variation_count <= 1:
                values = [value]
            elif isinstance(value, Integral):
                range_size = upper - lower + 1
                if range_size <= self._variation_count:
                    values = list(range(int(lower), int(upper) + 1))
                else:
                    step = (range_size - 1) / (self._variation_count - 1)
                    values = sorted(set(
                        lower + round(i * step) for i in range(self._variation_count)
                    ))
            elif lower == upper:
                values = [lower]
            else:
                step = (upper - lower) / (self._variation_count - 1)
                values = sorted(set(
                    lower + i * step for i in range(self._variation_count)
                ))

            for val in values:
                interventions.append({
                    'op': 'inject_value',
                    'param': param,
                    'value': val,
                    'reason': f"focus variation for {param}",
                })

        return interventions

    def _focus_bounds(self, center: int | float) -> tuple[int | float, int | float]:

        '''
        Compute [lower, upper] bounds for a numeric parameter value.

        Preserves int type when center is int (rounds bounds).

        '''

        if center == 0:
            half_width = self._focus_range_pct
        else:
            half_width = abs(center * self._focus_range_pct)

        lower = center - half_width
        upper = center + half_width

        if isinstance(center, Integral):
            lower = floor(lower)
            upper = ceil(upper)

        return lower, upper

    @staticmethod
    def _is_numeric(value: Any) -> bool:

        '''Check if a value is numeric (int or float, excluding bool).'''

        return isinstance(value, Real) and not isinstance(value, bool)

    @override
    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        return {
            'focus_active': self._focus_active,
            'breakthrough_combo': dict(self._breakthrough_combo) if self._breakthrough_combo is not None else None,
            'best_metric': self._best_metric,
            'rounds_since_improvement': self._rounds_since_improvement,
            'focused_params': list(self._focused_params),
        }

    @override
    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        required = (
            'focus_active',
            'breakthrough_combo',
            'best_metric',
            'rounds_since_improvement',
            'focused_params',
        )
        for key in required:
            if key not in state:
                raise ValueError(
                    f"Invalid FocusReducer state: missing required key '{key}'."
                )

        if state['focus_active']:
            if state['best_metric'] is None:
                raise ValueError(
                    "Invalid FocusReducer state: best_metric cannot be None when focus_active is True."
                )
            if state['breakthrough_combo'] is None:
                raise ValueError(
                    "Invalid FocusReducer state: breakthrough_combo cannot be None when focus_active is True."
                )

        self._focus_active = state['focus_active']
        self._breakthrough_combo = dict(state['breakthrough_combo']) if state['breakthrough_combo'] is not None else None
        self._best_metric = state['best_metric']
        self._rounds_since_improvement = state['rounds_since_improvement']
        self._focused_params = set(state['focused_params'])
