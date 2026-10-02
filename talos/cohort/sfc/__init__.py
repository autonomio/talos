"""Expose metric-diversity, Pareto and ranked cohort selection."""

from .all import select as select_all
from .diverse_metrics import select as select_diverse_metrics
from .pareto import select as select_pareto
from .top_n import select as select_top_n

BUILTIN_SELECTORS = {'all': select_all, 'top_n': select_top_n,
                     'pareto': select_pareto, 'diverse_metrics': select_diverse_metrics}
