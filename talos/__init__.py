from .scan.Scan import Scan
from .commands.reporting import Reporting
from .commands.predict import Predict
from .commands.deploy import Deploy
from .commands.evaluate import Evaluate
from. commands.restore import Restore

from .metrics.performance import Performance
from .examples import datasets, params
import astetik as plots
from kerasplotlib import TrainingLog as live

# del parameters, utils, scan
# del Performance, reporting, reducers, metrics, examples

__version__ = "0.4.3"
