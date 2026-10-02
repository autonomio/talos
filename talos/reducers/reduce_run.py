"""Dispatch legacy parameter reducers and preserve queue interventions."""

from .correlation import correlation
from .forrest import forrest
from .gamify import gamify
from .limit_by_metric import limit_by_metric
from .local_strategy import _controls, local_strategy
from .trees import trees

_REDUCERS = {'trees': trees, 'forrest': forrest}


def register_reducer(name, function):
    if not callable(function):
        raise TypeError('Reducer must be callable.')
    _REDUCERS[name] = function


def reduce_run(self):
    if self.performance_target is not None and limit_by_metric(self):
        self.param_object.param_index = []
    method = self.reduction_method
    if method is None:
        return self
    controls_before = _controls(self)
    if not hasattr(self, '_local_controls_baseline'):
        self._local_controls_baseline = controls_before
    pending_before = list(self.param_object.param_index)
    before = len(pending_before)
    if method == 'gamify':
        self = gamify(self)
    elif method == 'local_strategy':
        self = local_strategy(self)
    elif self.param_object.round_counter % self.reduction_interval == 0:
        if method in ('correlation', 'spearman', 'pearson', 'kendall'):
            self = correlation(self, 'spearman' if method == 'correlation' else method)
        elif callable(method):
            result = method(self)
            if isinstance(result, tuple) and len(result) == 2:
                self.param_object.remove_is(*result)
            elif result is not None and result is not False:
                self = result
        elif method in _REDUCERS:
            result = _REDUCERS[method](self)
            if isinstance(result, tuple) and len(result) == 2:
                self.param_object.remove_is(*result)
            elif result is not None and result is not False:
                self = result
        else:
            raise ValueError(f'Unknown reduction_method: {method!r}')
    if method != 'local_strategy' and hasattr(self.param_object, '_msq'):
        from talos.experiment.serialization import callable_reference, content_hash
        after = _controls(self)
        changed = {name: {'before': controls_before.get(name), 'after': after.get(name)}
                   for name in controls_before.keys() | after.keys() if content_hash(controls_before.get(name)) != content_hash(after.get(name))}
        if changed:
            self.param_object._msq._log_intervention('legacy_control_change',
                source='legacy_reducer', changes=changed,
                source_hash=callable_reference(method).get('source_hash') if callable(method) else None)
    if pending_before != self.param_object.param_index and hasattr(self.param_object, '_msq'):
        self.param_object._msq._log_intervention('legacy_pending_selection',
            source='legacy_reducer', reducer=method if isinstance(method, str) else 'custom',
            remaining_indexes=list(self.param_object.param_index),
            source_hash=getattr(self, '_local_strategy_hash', None))
    removed = max(0, before - len(self.param_object.param_index))
    if hasattr(self, 'pbar'):
        self.pbar.update(removed)
    return self
