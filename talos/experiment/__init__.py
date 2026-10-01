from .runner import RunResult, load_sfd, run

__all__ = ['RunResult', 'load_sfd', 'run', 'UniversalExperimentLoop']


def __getattr__(name):
    if name == 'UniversalExperimentLoop':
        from .experiment_core import UniversalExperimentLoop
        return UniversalExperimentLoop
    raise AttributeError(name)
