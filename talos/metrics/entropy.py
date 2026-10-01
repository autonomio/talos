def epoch_entropy(self, history):
    from talos.experiment.runner import _entropy
    values = _entropy(history)
    return [values.get(key, float('nan')) for key in self._metric_keys]
