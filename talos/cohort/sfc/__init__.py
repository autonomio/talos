from .all import select as select_all
from .top_n import select as select_top_n
from .pareto import select as select_pareto
from .diverse_metrics import select as select_diverse_metrics

BUILTIN_SELECTORS = {'all': select_all, 'top_n': select_top_n,
                     'pareto': select_pareto, 'diverse_metrics': select_diverse_metrics}
