import logging
from typing import Any

import polars as pl
from typing_extensions import override

from talos.experiment.reducer.filter_types import FILTER_SAMPLE
from talos.experiment.reducer.pruning_strategy import PruningStrategy

logger = logging.getLogger(__name__)


class SaturationReducer(PruningStrategy):

    '''
    Detect parameter values whose metric variance has collapsed and
    apply partial pruning via named filters.

    Uses Coefficient of Variation (CV = std / |mean|) over a rolling
    window of recent observations. Saturated values get a sample filter
    that retains retain_fraction of combinations for continued monitoring.

    '''

    def __init__(self,
                 *,
                 metric: str,
                 cv_threshold: float = 0.01,
                 window_size: int = 100,
                 min_samples_per_value: int = 20,
                 retain_fraction: float = 0.1,
                 active: bool = True) -> None:

        '''
        Initialize the SaturationReducer.

        Args:
            metric (str): Target metric column for variance analysis
            cv_threshold (float): CV below this indicates saturation
            window_size (int): Number of recent observations per value
            min_samples_per_value (int): Minimum samples before CV is computed
            retain_fraction (float): Fraction of combinations to keep for
                saturated values (0.0 to 1.0)
            active (bool): Whether this reducer is enabled

        '''

        if cv_threshold <= 0:
            raise ValueError(
                f"SaturationReducer cv_threshold must be > 0, got {cv_threshold}"
            )

        if not 0.0 <= retain_fraction <= 1.0:
            raise ValueError(
                f"SaturationReducer retain_fraction must be between 0.0 and 1.0, got {retain_fraction}"
            )

        if window_size <= 0:
            raise ValueError(
                f"SaturationReducer window_size must be > 0, got {window_size}"
            )

        if min_samples_per_value <= 0:
            raise ValueError(
                f"SaturationReducer min_samples_per_value must be > 0, got {min_samples_per_value}"
            )

        super().__init__(active=active)
        self._metric = metric
        self._cv_threshold = cv_threshold
        self._window_size = window_size
        self._min_samples_per_value = min_samples_per_value
        self._retain_fraction = retain_fraction
        self._saturated: set[tuple[str, Any]] = set()


    @override
    def analyze_and_intervene(self,
                              log: pl.DataFrame,
                              msq: Any) -> list[dict[str, Any]]:

        '''
        Analyze metric variance per parameter value and return interventions.

        Computes CV for each parameter value over a rolling window.
        Emits set_filter for saturated values and clear_filter for
        values that recover from saturation.

        '''

        if not self._active:
            return []

        df = log

        if df.is_empty() or self._metric not in df.columns:
            return []

        domain_params = msq.domain_keys
        currently_saturated: set[tuple[str, Any]] = set()
        interventions: list[dict[str, Any]] = []

        for param in domain_params:
            if param not in df.columns:
                continue

            grouped = (
                df.group_by(param, maintain_order=True)
                  .agg(
                      pl.col(self._metric).tail(self._window_size).std().alias('_std'),
                      pl.col(self._metric).tail(self._window_size).mean().alias('_mean'),
                      pl.col(self._metric).tail(self._window_size).len().alias('_count'),
                  )
            )

            for row in grouped.iter_rows(named=True):
                value = row[param]
                count = row['_count']
                mean = row['_mean']
                std = row['_std']

                if count < self._min_samples_per_value:
                    continue

                if mean is None or abs(mean) == 0 or std is None:
                    continue

                cv = std / abs(mean)

                if cv < self._cv_threshold:
                    currently_saturated.add((param, value))
                    if (param, value) not in self._saturated:
                        key = f"saturation_{param}_{value}"
                        interventions.append({
                            'op': 'set_filter',
                            'key': key,
                            'filter_type': FILTER_SAMPLE,
                            'filter_params': {
                                'param': param,
                                'value': value,
                                'fraction': self._retain_fraction,
                            },
                            'reason': f"saturated CV {cv:.4f} < {self._cv_threshold} for {param}={value}",
                        })

        for param, value in sorted(self._saturated - currently_saturated, key=lambda pv: (pv[0], repr(pv[1]))):
            key = f"saturation_{param}_{value}"
            interventions.append({
                'op': 'clear_filter',
                'key': key,
                'reason': f"no longer saturated for {param}={value}",
            })

        self._saturated = currently_saturated

        return interventions


    @override
    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        return {
            'saturated': list(self._saturated),
        }


    @override
    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        if 'saturated' not in state:
            raise ValueError("Invalid SaturationReducer state: missing required key 'saturated'.")

        self._saturated = set(tuple(x) for x in state['saturated'])
