from pathlib import Path
import inspect
import numbers
import numpy as np


class Trainer:
    """Explicit caller-SFD retraining with optional saved-trial metric verification."""
    def __init__(self, sfd=None, *, params=None, backend=None, model_factory=None,
                 result=None, run_dir=None):
        if result is None and run_dir is not None:
            from talos.experiment.runner import RunResult
            result = RunResult.load(run_dir)
        if sfd is None and result is not None:
            from talos.yaml.compiler import CompiledSFD
            sfd = CompiledSFD.from_run(result.run_dir)
        if isinstance(sfd, (str, Path)):
            from talos.experiment.runner import load_sfd
            sfd = load_sfd(sfd)
        if not all(callable(getattr(sfd, name, None)) for name in ('params', 'prep', 'model')):
            raise TypeError('Trainer requires a caller params/prep/model SFD')
        self.sfd, self.params, self.result = sfd, params, result
        self.backend, self.model_factory = backend, model_factory
        self.model = None
        self.history = None
        self.validation = {}

    def train(self, data=None, *, params=None, permutation_ids=None, validate_metrics=False,
              metric_rtol=0.01, metric_atol=1e-6):
        if permutation_ids is not None:
            if self.result is None:
                raise ValueError('Saved permutation selection requires an experiment result')
            from talos.inference import Sensor
            table = self.result.data
            column = '_trial_id' if '_trial_id' in table else 'id'
            members = []
            for identifier in permutation_ids:
                selected = table.loc[table[column].astype(str) == str(identifier)]
                if selected.empty:
                    raise ValueError(f'Unknown permutation {identifier}')
                row = selected.iloc[0]
                from talos.inference.params import trial_params
                chosen = trial_params(self.result, row, self.sfd.params())
                self._train_one(data, chosen)
                mismatches = self._validate_metrics(row, self.metrics, metric_rtol, metric_atol)
                self.validation[str(identifier)] = mismatches
                if validate_metrics and mismatches:
                    raise ValueError('Retrained metrics differ: ' + '; '.join(mismatches))
                members.append(Sensor(model=self.model, backend=self.backend, permutation_id=str(identifier),
                                      round_params=chosen))
            return members
        selected = params if params is not None else self.params
        if selected is None:
            selected = {key: values[0] for key, values in self.sfd.params().items()}
        return self._train_one(data, selected)

    def _train_one(self, data, selected):
        from talos.backends import normalise_result
        function = self.sfd.prep
        signature = inspect.signature(function)
        for args, kwargs in [((data, selected), {}), ((data,), {'round_params': selected}),
                             ((data,), {}), ((), {'round_params': selected}), ((), {})]:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            prepared = function(*args, **kwargs)
            break
        else:
            raise TypeError('Caller prep must accept context and optional round_params')
        output = normalise_result(self.sfd.model(prepared, selected), self.backend, self.model_factory)
        self.model, self.history = output['model'], output['history']
        self.metrics, self.round_params = output['metrics'], selected
        self.backend = output['backend']
        return self.model

    @staticmethod
    def _validate_metrics(original, metrics, rtol, atol):
        return [f'{key}: expected {original[key]}, got {value}' for key, value in metrics.items()
                if key in original and isinstance(value, numbers.Real) and isinstance(original[key], numbers.Real)
                and not np.isclose(value, original[key], rtol=rtol, atol=atol, equal_nan=True)]

    def predict(self, inputs, **options):
        from talos.backends import backend_for
        if self.model is None:
            raise RuntimeError('Call train() before predict()')
        return backend_for(self.model, self.backend).predict(self.model, inputs, **options)

    __call__ = train
