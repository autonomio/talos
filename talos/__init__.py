from talos.scan import Scan
from talos.reporting import Reporting
from talos.metrics.performance import Performance
from talos.examples import datasets, models, params
from talos.utils import save_load

del parameters, model, utils, scan, save_load,
del Performance, reporting, reducers, metrics, examples

__version__ = "0.1.8"
