import logging
import time
from typing import Any

import polars as pl
from typing_extensions import override

from talos.experiment.reducer.pruning_strategy import PruningStrategy

logger = logging.getLogger(__name__)


TRIM_RANDOM = 'random'
TRIM_WORST_FIRST = 'worst_first'


class BudgetReducer(PruningStrategy):

    '''
    Trim the search queue when projected completion exceeds resource
    budgets (walltime or permutation count).

    Purely resource-driven — no statistical analysis of results.
    Supports two trim strategies:
    - random: uniformly downsample permutations to the target count
    - worst_first: rank parameter values by mean metric performance
      and remove combinations containing the worst-performing values

    A random trim is always emitted as a fallback to ensure the queue
    is reduced to the target count if value-based pruning alone is
    insufficient.

    NOTE: Fires at most once per experiment — once trimming occurs,
    further calls to analyze_and_intervene return empty. The
    check_after_pct gate prevents premature trimming when throughput
    estimates are unreliable early in the experiment.

    '''

    def __init__(self,
                 *,
                 start_time: float | None = None,
                 max_walltime_hours: float | None = None,
                 max_permutations: int | None = None,
                 trim_strategy: str = 'random',
                 check_after_pct: float = 0.1,
                 metric: str | None = None,
                 maximize: bool = True,
                 active: bool = True) -> None:

        '''
        Initialize the BudgetReducer.

        Args:
            start_time (float | None): Experiment start time from time.time().
                Defaults to current time if not provided
            max_walltime_hours (float | None): Maximum allowed walltime in hours
            max_permutations (int | None): Maximum total permutations allowed
            trim_strategy (str): How to trim — 'random' or 'worst_first'
            check_after_pct (float): Only start checking after this fraction
                of budget is consumed (0.0 to 1.0)
            metric (str | None): Target metric column for worst_first strategy.
                Required when trim_strategy is 'worst_first'
            maximize (bool): If True, low metric values are "worst".
                If False, high metric values are "worst"
            active (bool): Whether this reducer is enabled

        '''

        if trim_strategy not in (TRIM_RANDOM, TRIM_WORST_FIRST):
            raise ValueError(
                f"BudgetReducer Unknown trim_strategy '{trim_strategy}'. Must be '{TRIM_RANDOM}' or '{TRIM_WORST_FIRST}'."
            )

        if trim_strategy == TRIM_WORST_FIRST and metric is None:
            raise ValueError(
                f"BudgetReducer metric is required when trim_strategy is '{TRIM_WORST_FIRST}'."
            )

        if max_walltime_hours is not None and max_walltime_hours <= 0:
            raise ValueError(
                f"BudgetReducer max_walltime_hours must be > 0, got {max_walltime_hours}"
            )

        if max_permutations is not None and max_permutations <= 0:
            raise ValueError(
                f"BudgetReducer max_permutations must be > 0, got {max_permutations}"
            )

        if not 0.0 <= check_after_pct <= 1.0:
            raise ValueError(
                f"BudgetReducer check_after_pct must be between 0.0 and 1.0, got {check_after_pct}"
            )

        super().__init__(active=active)
        self._start_time = start_time if start_time is not None else time.time()
        self._max_walltime_hours = max_walltime_hours
        self._max_permutations = max_permutations
        self._trim_strategy = trim_strategy
        self._check_after_pct = check_after_pct
        self._metric = metric
        self._maximize = maximize
        self._trimmed: bool = False


    @override
    def analyze_and_intervene(self,
                              log: pl.DataFrame,
                              msq: Any) -> list[dict[str, Any]]:

        '''
        Check resource budgets and trim the queue if over budget.

        Computes throughput from elapsed time and completed rounds,
        then projects whether the remaining queue can finish within
        the configured budget.

        '''

        if not self._active or self._trimmed:
            return []

        remaining = msq.remaining_count()
        if remaining is not None and remaining == 0:
            return []

        yielded = msq.yielded_count
        if yielded == 0:
            return []

        target = self._compute_target(yielded)
        if target is None or (remaining is not None and target >= remaining):
            return []

        new_total = yielded + target
        interventions: list[dict[str, Any]] = []

        if self._trim_strategy == TRIM_WORST_FIRST:
            worst_interventions = self._trim_worst_first(log, msq, target)
            interventions.extend(worst_interventions)

        interventions.append(self._trim_random(new_total))

        self._trimmed = True

        return interventions


    def _compute_target(self,
                        yielded: int) -> int | None:

        '''
        Compute how many more permutations the budget allows.

        Returns:
            int | None: Target remaining count, or None if no trimming needed

        '''

        target_from_walltime = None
        target_from_permutations = None

        if self._max_walltime_hours is not None:
            elapsed = time.time() - self._start_time
            elapsed_hours = elapsed / 3600.0
            budget_pct = elapsed_hours / self._max_walltime_hours

            if budget_pct >= self._check_after_pct:
                if elapsed_hours >= self._max_walltime_hours:
                    target_from_walltime = 0
                elif elapsed_hours > 0 and yielded > 0:
                    rate_per_hour = yielded / elapsed_hours
                    hours_left = self._max_walltime_hours - elapsed_hours
                    target_from_walltime = int(rate_per_hour * hours_left)

        if self._max_permutations is not None:
            budget_pct_perm = yielded / self._max_permutations

            if budget_pct_perm >= self._check_after_pct:
                if yielded >= self._max_permutations:
                    target_from_permutations = 0
                else:
                    target_from_permutations = self._max_permutations - yielded

        candidates = [t for t in [target_from_walltime, target_from_permutations] if t is not None]
        if not candidates:
            return None

        return min(candidates)


    def _trim_random(self, target_count: int) -> dict[str, Any]:

        '''Emit a trim intervention for random downsampling.'''

        return {
            'op': 'trim',
            'target_count': target_count,
            'reason': f"budget trim to {target_count} total permutations",
        }


    def _trim_worst_first(self,
                          log: pl.DataFrame,
                          msq: Any,
                          target_remaining: int) -> list[dict[str, Any]]:

        '''
        Emit remove_is interventions for worst-performing param values.

        Ranks completed parameter values by mean metric performance and
        removes combinations containing the worst-performing values
        until the remaining count is within budget.

        '''

        if log.is_empty() or self._metric not in log.columns:
            return []

        domain_keys = msq.domain_keys

        col = pl.col(self._metric)
        filter_expr = col.is_not_null()
        if log[self._metric].dtype.is_float():
            filter_expr = filter_expr & col.is_not_nan()
        filtered = log.filter(filter_expr)

        if filtered.is_empty():
            return []

        worst_values: list[tuple[str, Any]] = []

        for param in domain_keys:
            if param not in log.columns:
                continue

            grouped = (
                filtered
                   .group_by(param)
                   .agg(pl.col(self._metric).mean().alias('_mean'))
                   .sort('_mean', descending=not self._maximize)
            )

            if grouped.is_empty():
                continue

            worst_row = grouped.row(0, named=True)
            worst_values.append((param, worst_row[param]))

        if not worst_values:
            return []

        interventions: list[dict[str, Any]] = []
        remaining = msq.remaining_count() or 0
        full_dist = msq.distribution()
        domain_sizes = {p: len(full_dist.get(p, {})) for p in domain_keys}

        for param, value in worst_values:
            if remaining <= target_remaining:
                break

            n_values = domain_sizes.get(param, 0)
            if n_values <= 1:
                continue

            interventions.append({
                'op': 'remove_is',
                'param': param,
                'value': value,
                'reason': f"budget worst_first: removing worst-performing {param}={value}",
            })

            remaining -= remaining // n_values
            domain_sizes[param] = n_values - 1

        return interventions


    @override
    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        return {
            'start_time': self._start_time,
            'trimmed': self._trimmed,
        }


    @override
    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        required = ('start_time', 'trimmed')
        for key in required:
            if key not in state:
                raise ValueError(
                    f"Invalid BudgetReducer state: missing required key '{key}'."
                )

        self._start_time = state['start_time']
        self._trimmed = state['trimmed']
