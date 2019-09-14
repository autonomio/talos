import warnings
warnings.simplefilter('ignore')

# import commands
from .scan.Scan import Scan
from .commands.analyze import Analyze
from .commands.analyze import Analyze as Reporting
from .commands.predict import Predict
from .commands.deploy import Deploy
from .commands.evaluate import Evaluate
from .commands.restore import Restore

# import extras
from . import utils
from . import templates
from . import autom8

# the purpose of everything below is to keep the namespace completely clean

template_sub = [templates.datasets,
                templates.models,
                templates.params,
                templates.pipelines]

keep_from_templates = ['iris', 'cervical_cancer', 'titanic', 'breast_cancer',
                       'icu_mortality', 'telco_churn', 'mnist']

for sub in template_sub:
    for key in list(sub.__dict__):
        if key.startswith('__') is False:
            if key not in keep_from_templates:
                delattr(sub, key)

del commands, scan, model, metrics, key
del sub, keep_from_templates, template_sub, warnings

__version__ = "0.6.4"
