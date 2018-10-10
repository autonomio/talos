from .scan.Scan import Scan
from .commands.reporting import Reporting
from .commands.predict import Predict
from .commands.deploy import Deploy
from .commands.evaluate import Evaluate
from. commands.restore import Restore

from .examples import datasets, params
import astetik as plots
from kerasplotlib import TrainingLog as live

__version__ = "0.4.4"
