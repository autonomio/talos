"""Candidate selection and model recovery for old and new Talos runs."""
from pathlib import Path

from talos.backends import backend_for


def best_model(self, metric, asc):
    values = self.data[metric].dropna()
    if values.empty:
        raise ValueError('No candidate has a usable value for ' + metric)
    return self.data.loc[values.index].sort_values(metric, ascending=asc, kind='stable').iloc[0].name


def activate_model(self, model_id, saved=False, custom_objects=None, model_factory=None):
    custom_objects = custom_objects if custom_objects is not None else getattr(self, 'custom_objects', None)
    model_factory = model_factory if model_factory is not None else getattr(self, 'model_factory', None)
    models = getattr(self, 'models', None)
    if models is not None and not saved:
        model = models.get(model_id) if isinstance(models, dict) else models[model_id]
        if model is not None:
            return model
    artifacts = getattr(self, 'artifacts', None)
    if artifacts is not None:
        descriptor = artifacts.get(model_id) if isinstance(artifacts, dict) else artifacts[model_id]
        if descriptor:
            return backend_for(backend=descriptor['backend']).load(descriptor, custom_objects, model_factory)
    if hasattr(self, 'model') and not hasattr(self, 'saved_models'):
        return self.model
    if saved:
        details = self.details
        path = Path(str(details['experiment_name'])) / str(details['experiment_id']) / str(model_id)
        adapter = backend_for(backend=getattr(self, 'backend', 'tensorflow'))
        return adapter.load({'backend': adapter.name, 'format': 'keras', 'path': str(path)}, custom_objects)
    saved_models = getattr(self, 'saved_models', [])
    saved_weights = getattr(self, 'saved_weights', [])
    if model_id >= len(saved_models) or model_id >= len(saved_weights) or saved_weights[model_id] is None:
        raise ValueError('This run did not retain the selected model; enable save_weights or save_models.')
    import tensorflow as tf
    model = tf.keras.models.model_from_json(saved_models[model_id], custom_objects=custom_objects)
    model.set_weights(saved_weights[model_id])
    return model
