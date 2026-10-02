"""Enumerate parameter combinations in deterministic permuted order."""

from __future__ import annotations

import math
import random
from typing import Any

from typing_extensions import override

from talos.experiment.param_domain import ParamDomain
from talos.experiment.param_search.search_strategy import SearchStrategy

GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


class GridStrategy(SearchStrategy):

    '''
    Exhaustive enumeration of all parameter combinations.

    Uses index-to-combination modular arithmetic for O(1) generation
    per sample. Optional shuffle reorders the enumeration via LCG
    (linear congruential generator) for better early coverage.
    Memory grows with yielded combos due to _seen tracking, which
    is acceptable because Grid targets small search spaces.

    '''

    def __init__(self,
                 domain: ParamDomain, *,
                 seed: int | None = None,
                 shuffle: bool = False) -> None:

        super().__init__(domain, seed=seed)
        self._shuffle = shuffle
        self._rebuild()

    @property
    @override
    def is_finite(self) -> bool:

        return True

    def _rebuild(self) -> None:

        '''Recompute keys, sizes, total, and LCG params from current domain.'''

        params = self._domain.params
        self._keys = sorted(params.keys())
        self._values = [params[k] for k in self._keys]
        self._sizes = [len(v) for v in self._values]
        self._total = math.prod(self._sizes)
        self._current_index = 0
        self._generated_count = 0

        if self._shuffle and self._total > 0:
            self._lcg_multiplier, self._lcg_increment = _lcg_params(
                self._total, self._seed,
            )

    def _permute_index(self, logical_index: int) -> int:

        if not self._shuffle:
            return logical_index

        return _lcg_map(
            logical_index, self._total,
            self._lcg_multiplier, self._lcg_increment,
        )

    def _index_to_combo(self, index: int) -> dict[str, Any]:

        '''Convert flat index to parameter combination via modular arithmetic.'''

        combo: dict[str, Any] = {}
        remaining = index
        for i, key in enumerate(self._keys):
            combo[key] = self._values[i][remaining % self._sizes[i]]
            remaining //= self._sizes[i]
        return combo

    @override
    def __next__(self) -> dict[str, Any]:

        while self._current_index < self._total:
            physical = self._permute_index(self._current_index)
            combo = self._index_to_combo(physical)
            self._current_index += 1
            if self._is_unseen(combo):
                self._generated_count += 1
                return combo

        raise StopIteration

    @override
    def on_domain_changed(
        self, _domain: ParamDomain, _changed_params: list[str],
    ) -> None:

        self._rebuild()

    @override
    def get_state(self) -> dict[str, Any]:

        return {
            'current_index': self._current_index,
            'generated_count': self._generated_count,
            'shuffle': self._shuffle,
            'seed': self._seed,
        }

    @override
    def set_state(self, state: dict[str, Any]) -> None:

        saved_shuffle = state.get('shuffle')
        saved_seed = state.get('seed')
        if saved_shuffle is not None and saved_shuffle != self._shuffle:
            raise ValueError(
                f"GridStrategy Checkpoint shuffle={saved_shuffle} does not match current shuffle={self._shuffle}."
            )
        if saved_seed is not None and saved_seed != self._seed:
            raise ValueError(
                f"GridStrategy Checkpoint seed={saved_seed} does not match current seed={self._seed}."
            )
        self._current_index = state['current_index']
        self._generated_count = state['generated_count']


def _lcg_params(n: int, seed: int | None) -> tuple[int, int]:

    '''
    Compute LCG parameters for a full-period permutation of [0, n).

    Finds a multiplier coprime to n and a seeded increment for
    a linear congruential generator that visits every index exactly once.

    Args:
        n (int): Size of the space to permute
        seed (int | None): Seed for choosing increment

    Returns:
        tuple[int, int]: (multiplier, increment) for the LCG
    '''

    multiplier = _find_coprime(n)
    rng = random.Random(seed)
    increment = rng.randint(0, n - 1)
    return multiplier, increment


def _find_coprime(n: int) -> int:

    '''Find a value coprime to n that provides good mixing.'''

    candidate = max(2, int(n * GOLDEN_RATIO_CONJUGATE))
    while math.gcd(candidate, n) != 1:
        candidate += 1
    return candidate


def _lcg_map(logical_index: int, n: int,
             multiplier: int, increment: int) -> int:

    '''Map logical index to physical index via LCG.'''

    return (multiplier * logical_index + increment) % n
