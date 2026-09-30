from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


ACTION_SUGGEST = 'suggest'


class PruningStrategy(ABC):

    '''
    Abstract base for all pruning strategies (reducers).

    Reducers are pure analysis components. They receive an experiment log
    and an MSQ reference (for observability only), analyze the data, and
    return a list of interventions communicated to MSQ for modifying the
    param space.

    '''

    def __init__(self, *, active: bool = True) -> None:

        '''
        Initialize the PruningStrategy.

        Args:
            active (bool): Whether this reducer is enabled. Inactive reducers
                return empty intervention lists from analyze_and_intervene

        '''

        super().__init__()

        self._active = active


    @abstractmethod
    def analyze_and_intervene(self,
                              log: Any,
                              msq: Any) -> list[dict[str, Any]]:

        '''
        Analyze experiment log and return intervention dicts.

        Args:
            log (Any): Current experiment log with results so far
            msq (Any): MSQ reference for observability (remaining_count,
                distribution). Must NOT be mutated

        Returns:
            list[dict[str, Any]]: Intervention dicts. Empty list if no pruning needed

        '''

        ...


    @abstractmethod
    def get_state(self) -> dict[str, Any]:

        '''Export state for checkpointing.'''

        ...


    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:

        '''Restore state from checkpoint.'''

        ...


    @property
    def active(self) -> bool:

        return self._active


    @active.setter
    def active(self, value: bool) -> None:

        self._active = value
