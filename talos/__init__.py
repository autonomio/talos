from importlib import import_module

__version__ = '2.0.6'

_EXPORTS = {
    'Sensor': ('talos.inference', 'Sensor'),
    'Trainer': ('talos.inference', 'Trainer'),
    'Cohort': ('talos.cohort', 'Cohort'),
    'Log': ('talos.log', 'Log'),
    'MLManifest': ('talos.experiment.manifest_core', 'MLManifest'),
    'Scan': ('talos.scan.Scan', 'Scan'),
    'Analyze': ('talos.commands.analyze', 'Analyze'),
    'Reporting': ('talos.commands.analyze', 'Analyze'),
    'Predict': ('talos.commands.predict', 'Predict'),
    'Evaluate': ('talos.commands.evaluate', 'Evaluate'),
    'Deploy': ('talos.commands.deploy', 'Deploy'),
    'Restore': ('talos.commands.restore', 'Restore'),
    'run': ('talos.experiment.runner', 'run'),
    'RunResult': ('talos.experiment.runner', 'RunResult'),
    'UniversalExperimentLoop': ('talos.experiment.experiment_core', 'UniversalExperimentLoop'),
}
_MODULES = {'utils', 'templates', 'autom8', 'callbacks', 'experiment', 'backends', 'scalers', 'transforms', 'metrics', 'calibration', 'preparation', 'log', 'inference', 'cohort', 'sfd', 'yaml'}
__all__ = [*_EXPORTS, *_MODULES, '__version__']


def __getattr__(name):
    if name in _EXPORTS:
        module, symbol = _EXPORTS[name]
        value = getattr(import_module(module), symbol)
    elif name in _MODULES:
        value = import_module('talos.' + name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
