from contextvars import ContextVar

_current_trial = ContextVar('talos_trial', default=None)


def get_trial_context():
    return _current_trial.get()


def set_trial_context(context):
    return _current_trial.set(context)


def reset_trial_context(token):
    _current_trial.reset(token)
