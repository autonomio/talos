"""Load backend-specific model artifacts with caller restoration options."""

from pathlib import Path

from talos.backends import backend_for


def load_model(saved_model, custom_objects=None, backend='tensorflow', model_factory=None):
    if isinstance(saved_model, dict):
        return backend_for(backend=saved_model['backend']).load(saved_model, custom_objects, model_factory)
    path = Path(saved_model)
    if path.suffix in ('.keras', '.h5'):
        descriptor = {'backend': backend, 'format': 'keras', 'path': str(path)}
    else:
        descriptor = {'backend': backend, 'format': 'keras_json_weights', 'path': str(path) + '.json', 'weights': str(path) + '.h5'}
    return backend_for(backend=backend).load(descriptor, custom_objects, model_factory)
