"""Sample unique parameter combinations with reproducible state."""

from __future__ import annotations

import random
from typing import Any, cast

from typing_extensions import override

from talos.experiment.param_domain import ParamDomain
from talos.experiment.param_search.search_strategy import SearchStrategy

MAX_DEDUP_RETRIES = 1000


class RandomStrategy(SearchStrategy):

    '''
    Lazy random sampling from the parameter domain.

    Samples each parameter independently via uniform random choice.
    Stops when the finite domain is exhausted. Uses _is_unseen for
    dedup to avoid repeating previously seen combinations.

    '''

    def __init__(self, domain: ParamDomain, *, seed: int | None = None) -> None:

        super().__init__(domain, seed=seed)
        self._rng = random.Random(seed)
        self._refresh_cache()

    def _refresh_cache(self) -> None:

        '''Snapshot sorted keys and values from domain to avoid repeated defensive copies.'''

        params = self._domain.params
        self._stable_keys = sorted(params)
        self._cached_values = {k: params[k] for k in self._stable_keys}

    @override
    def on_domain_changed(
        self, _domain: ParamDomain, _changed_params: list[str],
    ) -> None:

        self._refresh_cache()

    @override
    def __next__(self) -> dict[str, Any]:

        if self._domain.total_combinations == 0:
            raise StopIteration
        for _ in range(MAX_DEDUP_RETRIES):
            combo = {k: self._rng.choice(self._cached_values[k]) for k in self._stable_keys}
            if self._is_unseen(combo):
                self._generated_count += 1
                return combo

        # Finite domains must finish honestly even near coupon-collector exhaustion.
        import itertools
        for values in itertools.product(*(self._cached_values[key] for key in self._stable_keys)):
            combo = dict(zip(self._stable_keys, values))
            if self._is_unseen(combo):
                self._generated_count += 1
                return combo
        raise StopIteration

    @override
    def get_state(self) -> dict[str, Any]:

        return {
            'rng_state': self._rng.getstate(),
            'generated_count': self._generated_count,
        }

    @override
    def set_state(self, state: dict[str, Any]) -> None:

        rng_state = state['rng_state']
        # JSON serialization converts tuples to lists; restore tuple structure
        if isinstance(rng_state, list):
            rng_list = cast(list[Any], rng_state)
            rng_state = (rng_list[0], tuple(cast(list[Any], rng_list[1])), rng_list[2])
        self._rng.setstate(rng_state)
        self._generated_count = state['generated_count']
