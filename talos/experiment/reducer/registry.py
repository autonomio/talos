from talos.experiment.reducer.budget_reducer import BudgetReducer
from talos.experiment.reducer.correlation_reducer import CorrelationReducer
from talos.experiment.reducer.focus_reducer import FocusReducer
from talos.experiment.reducer.pruning_strategy import PruningStrategy
from talos.experiment.reducer.sanity_reducer import SanityReducer
from talos.experiment.reducer.saturation_reducer import SaturationReducer

REDUCER_REGISTRY: dict[str, type[PruningStrategy]] = {
    'budget': BudgetReducer,
    'correlation': CorrelationReducer,
    'focus': FocusReducer,
    'sanity': SanityReducer,
    'saturation': SaturationReducer,
}
