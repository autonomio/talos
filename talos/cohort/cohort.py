"""Select parameter cohorts using caller experiment metrics."""

import inspect
from pathlib import Path

import numpy as np

from talos.cohort.sfc import BUILTIN_SELECTORS


class Cohort:
    """Select saved trials and aggregate compatible outputs for caller tasks."""
    def __init__(self, *, experiment_id=None, experiment_log_path=None, result=None,
                 permutation_ids=None, selector=None, selector_params=None,
                 aggregation='mean', task='regression', weights=None, threshold=0.5,
                 custom_objects=None, model_factory=None):
        from talos.experiment.runner import RunResult
        if result is None:
            source = experiment_log_path if experiment_log_path is not None else experiment_id
            if source is None:
                raise ValueError('Supply an experiment result or directory')
            result = RunResult.load(Path(source), custom_objects=custom_objects, model_factory=model_factory)
        if permutation_ids is not None and selector is not None:
            raise ValueError('Supply permutation_ids or a selector, not both')
        self.result = result
        self.metadata = getattr(result, 'metadata', {})
        self.experiment_dir = getattr(result, 'run_dir', None)
        self.aggregation_mode, self.task = aggregation, task
        self.weights, self.threshold = weights, threshold
        table = result.data.copy()
        if 'id' not in table:
            table['id'] = table['_trial_id']
        self.available_permutation_ids = [str(value) for value in table['id']]
        if permutation_ids is None:
            import polars as pl

            from talos.experiment.serialization import dumps

            def selection_value(value):
                if isinstance(value, np.generic):
                    value = value.item()
                return value if value is None or isinstance(value, (str, int, float, bool)) else dumps(value)
            rows = [{key: selection_value(value) for key, value in row.items()}
                    for row in table.to_dict(orient='records')]
            context = {'results': pl.from_dicts(rows, strict=False, infer_schema_length=None), 'metadata': self.metadata,
                       'experiment_dir': self.experiment_dir,
                       'available_permutation_ids': self.available_permutation_ids,
                       'round_entries': getattr(result, 'round_history', [])}
            function = BUILTIN_SELECTORS[selector or 'all'] if isinstance(selector, (str, type(None))) else selector
            permutation_ids = function(context, **(selector_params or {}))
        selected = [str(value) for value in permutation_ids]
        if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(self.available_permutation_ids):
            raise ValueError('Selection must contain unique known permutation ids')
        self.permutation_ids = selected
        self.custom_objects, self.model_factory = custom_objects, model_factory
        self._members = []
        from talos.experiment.serialization import content_hash
        reference = self.metadata.get('yaml_reference') or {}
        self.manifest_id = reference.get('manifest_id')
        self.cohort_id = 'sha256:' + content_hash({
            'ids': sorted(selected), 'aggregation': aggregation, 'task': task, 'weights': weights,
            'manifest': self.manifest_id, 'run_identity': result.details.get('identity_hash')})

    def set_members(self, members):
        by_id = {str(member.permutation_id): member for member in members}
        if len(by_id) != len(members) or set(by_id) != set(self.permutation_ids):
            raise ValueError('Members must match selected permutation ids exactly')
        self._members = [by_id[identifier] for identifier in self.permutation_ids]

    def predict(self, inputs, **options):
        if not self._members:
            from talos.inference import Sensor
            self._members = [Sensor(result=self.result, permutation_id=identifier,
                                    custom_objects=self.custom_objects, model_factory=self.model_factory)
                             for identifier in self.permutation_ids]
        predictions = [member.predict(inputs, **options) for member in self._members]
        return self._aggregate_outputs(predictions)

    def _aggregate_outputs(self, predictions):
        if isinstance(predictions[0], dict):
            keys = set(predictions[0])
            if any(not isinstance(value, dict) or set(value) != keys for value in predictions):
                raise ValueError('Cohort output dictionaries must have matching keys')
            return {key: self._aggregate_outputs([value[key] for value in predictions]) for key in predictions[0]}
        if isinstance(predictions[0], tuple):
            if any(not isinstance(value, tuple) or len(value) != len(predictions[0]) for value in predictions):
                raise ValueError('Cohort output tuples must have matching lengths')
            return tuple(self._aggregate_outputs([value[index] for value in predictions]) for index in range(len(predictions[0])))
        predictions = [np.asarray(value) for value in predictions]
        if len({array.shape for array in predictions}) != 1:
            raise ValueError('Cohort prediction shapes must agree')
        stacked = np.stack(predictions)
        if callable(self.aggregation_mode):
            signature = inspect.signature(self.aggregation_mode)
            try:
                signature.bind(stacked, weights=self.weights)
            except TypeError:
                return self.aggregation_mode(stacked)
            return self.aggregation_mode(stacked, weights=self.weights)
        if self.aggregation_mode in ('mean', 'probability_weighted'):
            return np.average(stacked, axis=0, weights=self.weights)
        if self.aggregation_mode in ('vote', 'majority_vote'):
            labels = stacked
            if self.task == 'multiclass' and stacked.ndim >= 3:
                labels = stacked.argmax(axis=-1)
            elif self.task == 'binary':
                labels = (stacked >= self.threshold).astype(int)
            return np.apply_along_axis(lambda column: np.unique(column, return_counts=True)[0][
                np.argmax(np.unique(column, return_counts=True)[1])], 0, labels)
        raise ValueError('Aggregation must be mean, vote, or a caller callable')

    predict_all = predict
    __call__ = predict
