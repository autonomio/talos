# import commands
from .scan.Scan import Scan
from .commands.reporting import Reporting
from .commands.predict import Predict
from .commands.deploy import Deploy
from .commands.evaluate import Evaluate
from .commands.restore import Restore
from .commands.autom8 import Autom8

# other internal imports
from .examples import datasets, params

# external append imports
import sklearn.metrics as performance

from .utils.connection_check import is_connected

if is_connected() is True:
	import astetik as plots
else:
	print("NO INTERNET CONNECTION: Reporting plots will not work.")
	
from kerasplotlib import TrainingLog as live

__version__ = "0.4.6"
