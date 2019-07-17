# import commands
from .scan.Scan import Scan
from .commands.reporting import Reporting
from .commands.predict import Predict
from .commands.deploy import Deploy
from .commands.evaluate import Evaluate
from .commands.restore import Restore
from .commands.autom8 import Autom8
from .commands.params import Params
from .commands.kerasmodel import KerasModel
from . import utils
from . import templates

# the purpose of everything below is to keep the namespace completely clean

template_sub = [templates.datasets,
                templates.models,
                templates.params,
                templates.pipelines]

keep_from_templates = ['iris', 'cervical_cancer', 'titanic', 'breast_cancer',
                       'icu_mortality']

for sub in template_sub:
    for key in list(sub.__dict__):
        if key.startswith('__') is False:
            if key not in keep_from_templates:
                delattr(sub, key)

del commands, scan, model, metrics, key
del sub, keep_from_templates, template_sub

__version__ = "0.6.0"
