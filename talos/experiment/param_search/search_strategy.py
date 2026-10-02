"""Define checkpointed parameter search and feedback contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, cast

from talos.experiment.param_domain import DomainObserver, ParamDomain
from talos.experiment.serialization import content_hash


class SearchStrategy(ABC):

    '''
    Abstract base for all search strategies.

    Strategies are iterators that yield dict[str, Any] combination of parameters.
    They hold a reference to a ParamDomain and register as observers
    so they can react to domain mutations.

    '''

    def __init__(self, domain: ParamDomain, *, seed: int | None = None) -> None:

        '''
        Initialize the SearchStrategy.

        Args:
            domain (ParamDomain): ParamDomain to generate combinations from
            seed (int | None): Optional random seed for reproducibility

        '''

        super().__init__()

        self._domain = domain
        self._seed = seed
        self._generated_count: int = 0  # subclasses must increment in __next__
        self._seen: set[str] = set()
        self._domain.add_observer(cast(DomainObserver, self))

    def __iter__(self) -> Iterator[dict[str, Any]]:

        return self

    @abstractmethod
    def __next__(self) -> dict[str, Any]:

        '''Return next parameter combination. Raise StopIteration when exhausted.'''

        ...

    def on_domain_changed(
        self, _domain: ParamDomain, _changed_params: list[str],
    ) -> None:

        '''Called when ParamDomain is mutated.

        Override if strategy maintains state dependent on the domain.
        '''

        return

    def update_from_feedback(
        self, _log: Any, _interventions: list[dict[str, Any]],
    ) -> None:

        '''Hook for strategies that adapt based on experiment feedback.

        Default is no-op. Stateful strategies (e.g. TPE) override this.
        '''

        return

    @property
    def domain(self) -> ParamDomain:

        return self._domain

    @property
    def generated_count(self) -> int:

        return self._generated_count

    @property
    def is_finite(self) -> bool:

        '''Whether this strategy has a finite number of combinations.

        Override to return True for exhaustive strategies (e.g. Grid).
        '''

        return False

    def compute_param_hash(self, combo: dict[str, Any]) -> str:

        '''
        Compute deterministic hash of a parameter combination.

        Returns the full 64 hex characters (256 bits) of the SHA-256
        digest.

        NOTE:
            Excludes the explicit reserved MSQ metadata columns. User
            parameter names beginning with underscores remain part of
            the parameter identity.

        Args:
            combo (dict[str, Any]): Parameter combination or log row

        Returns:
            str: 64-character hex digest
        '''

        metadata = {'_id', '_trial_id', '_param_hash', '_round_index', '_injected', '_generation_index', '_search_strategy'}
        clean = {k: v for k, v in combo.items() if k not in metadata}
        return content_hash(clean)

    def mark_seen(self, combo: dict[str, Any]) -> str:

        '''
        Register a combination as seen and return its hash.

        Called by MSQ after a combo passes filters and is yielded.
        Reuses ``_id`` if already attached by ``_is_unseen``.

        Args:
            combo (dict[str, Any]): Parameter combination

        Returns:
            str: The param hash
        '''

        h = combo.get('_id') or self.compute_param_hash(combo)
        self._seen.add(h)
        return h

    def _is_unseen(self, combo: dict[str, Any]) -> bool:

        '''
        Check if a combination has been seen before.

        Computes the hash, checks against _seen set, and attaches
        ``_id`` to the combo for downstream reuse. Does not
        register — registration is deferred to ``mark_seen`` so that
        combos rejected by MSQ filters remain available.

        Args:
            combo (dict[str, Any]): Parameter combination to check

        Returns:
            bool: True if novel (not seen before)
        '''

        h = self.compute_param_hash(combo)
        if h in self._seen:
            return False
        combo['_id'] = h
        return True

    def rebuild_seen_from_log(self, hashes: Iterable[str]) -> None:

        '''Populate _seen from an iterable of hashes.

        Used on resume to reconstruct dedup state from the experiment log.

        Args:
            hashes (Iterable[str]): Hash strings from previous runs
        '''

        self._seen = set(hashes)

    @abstractmethod
    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        ...

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        ...
