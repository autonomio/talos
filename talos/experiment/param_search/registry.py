from __future__ import annotations

from talos.experiment.param_search.grid_strategy import GridStrategy
from talos.experiment.param_search.random_strategy import RandomStrategy
from talos.experiment.param_search.search_strategy import SearchStrategy

STRATEGY_REGISTRY: dict[str, type[SearchStrategy]] = {
    'random': RandomStrategy,
    'grid': GridStrategy,
}
