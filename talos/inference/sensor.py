"""Build inference inputs and select stored trained models."""

from pathlib import Path

from talos.inference.errors import ArtifactError


class Sensor:
    """Restore and predict from a trained artifact; no data reading or retraining."""
    def __init__(self, experiment_log_path=None, permutation_id=None, *, run_dir=None,
                 result=None, metric=None, asc=None, custom_objects=None,
                 model_factory=None, preprocess=None, model=None, backend=None, round_params=None):
        from talos.experiment.runner import RunResult
        if result is None and model is None:
            source = run_dir if run_dir is not None else experiment_log_path
            if source is None:
                raise ValueError('Supply a result or saved run directory')
            result = RunResult.load(Path(source), custom_objects=custom_objects, model_factory=model_factory)
        self.result = result
        self.run_dir = getattr(result, 'run_dir', None)
        self.permutation_id = permutation_id
        self.metric, self.asc = metric, asc
        self.preprocess = preprocess
        self.custom_objects, self.model_factory = custom_objects, model_factory
        self.metadata = getattr(result, 'metadata', {})
        self.backend = backend
        self._round_params = round_params
        reference = self.metadata.get('yaml_reference') or {}
        self.manifest_id = reference.get('manifest_id', self.metadata.get('manifest_id'))
        self._model = model

    @property
    def model(self):
        if self._model is None:
            if self.permutation_id is None:
                self._model = self.result.best_model(metric=self.metric, asc=self.asc, custom_objects=self.custom_objects, model_factory=self.model_factory)
            else:
                from talos.backends import backend_for
                artifacts = self.result.artifacts
                column = '_trial_id' if '_trial_id' in self.result.data else 'id'
                identifiers = [str(identifier) for identifier in self.result.data[column]]
                if str(self.permutation_id) not in identifiers:
                    raise ArtifactError(f'Unknown permutation {self.permutation_id}')
                descriptor = artifacts[identifiers.index(str(self.permutation_id))]
                if descriptor is None:
                    raise ArtifactError(f'No saved model for permutation {self.permutation_id}')
                self._model = backend_for(backend=descriptor.get('backend')).load(
                    descriptor, custom_objects=self.custom_objects, model_factory=self.model_factory)
        return self._model

    @property
    def round_params(self):
        if self._round_params is not None:
            return dict(self._round_params)
        if self.result is None:
            return {}
        table = self.result.data
        if self.permutation_id is not None:
            column = '_trial_id' if '_trial_id' in table else 'id'
            selected = table.loc[table[column].astype(str) == str(self.permutation_id)]
            if selected.empty:
                raise ArtifactError(f'Unknown permutation {self.permutation_id}')
            row = selected.iloc[0]
        else:
            metric, ascending = self.result._objective(self.metric, self.asc)
            row = table.dropna(subset=[metric]).sort_values(metric, ascending=ascending, kind='stable').iloc[0]
        from talos.inference.params import trial_params
        return trial_params(self.result, row)

    def predict(self, inputs, **options):
        from talos.backends import backend_for
        prepared = self.preprocess(inputs) if self.preprocess is not None else inputs
        return backend_for(self.model, self.backend).predict(self.model, prepared, **options)

    predict_all = predict
    __call__ = predict
