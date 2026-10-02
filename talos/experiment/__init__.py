"""Expose parameter-sweep execution, results and prepared-data manifests."""

from .runner import RunResult, load_sfd, run

__all__ = ['RunResult', 'UniversalExperimentLoop', 'load_sfd', 'run']


def __getattr__(name):
    if name == 'UniversalExperimentLoop':
        from .experiment_core import UniversalExperimentLoop
        return UniversalExperimentLoop
    raise AttributeError(name)
