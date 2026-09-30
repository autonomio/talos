"""Framework imports occur only when an adapter is used."""
import gc
import importlib
import inspect
import sys
import copy
import numbers
import json
import tempfile
import hashlib
from pathlib import Path
from collections.abc import Mapping


def reference(value):
    if isinstance(value, str):
        return value
    module = getattr(value, '__module__', None)
    name = getattr(value, '__qualname__', None)
    if module and name and '<locals>' not in name and module != '__main__':
        return module + ':' + name
    return None


def resolve(value, source=None):
    if not isinstance(value, str):
        return value
    if ':' in value:
        module, name = value.split(':', 1)
    else:
        module, _, name = value.rpartition('.')
    if not module:
        raise ValueError('Use an importable module:qualified_name reference.')
    existing = sys.modules.get(module)
    existing_source = getattr(existing, '__file__', None)
    same_source = (existing_source is not None and source is not None
                   and Path(existing_source).is_file() and Path(source).is_file()
                   and _digest(existing_source) == _digest(source))
    if source is not None and not same_source:
        specification = importlib.util.spec_from_file_location(module, source)
        if specification is None or specification.loader is None:
            raise ImportError('Cannot load model factory source ' + str(source))
        loaded = importlib.util.module_from_spec(specification)
        sys.modules[module] = loaded
        specification.loader.exec_module(loaded)
    out = importlib.import_module(module)
    for part in name.split('.'):
        out = getattr(out, part)
    return out


def _number(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            pass
    if hasattr(value, 'numpy'):
        value = value.numpy()
        if hasattr(value, 'item'):
            try:
                value = value.item()
            except (ValueError, RuntimeError):
                pass
    if isinstance(value, numbers.Real) and not isinstance(value, (int, float, bool)):
        return float(value)
    return value


def _normalise_history(history):
    normalized = {}
    for key, values in history.items():
        if not isinstance(key, str):
            raise TypeError('History metric names must be strings: ' + str(key))
        if isinstance(values, (str, bytes, dict)):
            raise TypeError('History ' + repr(key) + ' must contain a sequence of real numeric scalars.')
        try:
            series = iter(values)
        except TypeError as error:
            raise TypeError('History ' + repr(key) + ' must contain a sequence of real numeric scalars.') from error
        points = []
        for index, value in enumerate(series):
            description = f'History {key}[{index}] must be a real numeric scalar.'
            try:
                point = _number(value)
            except (TypeError, ValueError, RuntimeError) as error:
                raise TypeError(description) from error
            if not isinstance(point, numbers.Real):
                raise TypeError(description)
            points.append(point)
        normalized[key] = points
    return normalized


def normalise_result(output, backend=None, model_factory=None):
    """Normalize a legacy callback or SFD result without owning its training."""
    model = None
    predictions = None
    if isinstance(output, tuple) and len(output) == 2:
        history, model = output
        # Original Talos Torch examples return (module_with_history, parameters()).
        if hasattr(history, 'state_dict') and not hasattr(model, 'state_dict'):
            model = history
        history = getattr(history, 'history', history)
        if not isinstance(history, dict):
            raise TypeError('The first model return must expose a history dictionary.')
        history = _normalise_history(history)
        metrics = {key: _number(values[-1]) for key, values in history.items() if len(values)}
    elif isinstance(output, dict):
        model = output.get('_model', output.get('model'))
        predictions = output.get('_preds', output.get('predictions'))
        history = output.get('_history', output.get('history', {}))
        history = getattr(history, 'history', history)
        if history is None:
            history = {}
        if not isinstance(history, dict):
            raise TypeError('SFD history must be a dictionary or a history object.')
        history = _normalise_history(history)
        metrics = output.get('metrics')
        if metrics is None:
            reserved = {'model', 'history', 'predictions', 'backend', 'factory', 'model_factory', 'metrics'}
            metrics = {key: value for key, value in output.items()
                       if not key.startswith('_') and key not in reserved}
        metrics = {key: _number(value) for key, value in metrics.items()}
        for key, values in history.items():
            if len(values):
                metrics.setdefault(key, _number(values[-1]))
        backend = output.get('backend', backend)
        model_factory = output.get('model_factory', output.get('factory', model_factory))
    else:
        raise TypeError('Return (history, model) or an SFD metric dictionary.')
    for key, value in metrics.items():
        if not isinstance(key, str) or not isinstance(value, numbers.Real):
            raise TypeError('Trial metrics must have string names and numeric scalar values: ' + str(key))
    name = backend_for(model, backend).name if model is not None else (backend or 'generic')
    out = {'metrics': metrics, 'history': history, 'model': model, 'backend': name}
    if predictions is not None:
        out['predictions'] = predictions
    if model_factory is not None:
        out['factory'] = model_factory
    return out


def _digest(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify(descriptor):
    for field, checksum in [('path', 'sha256'), ('factory_source', 'factory_source_sha256')]:
        if descriptor.get(checksum) and _digest(descriptor[field]) != descriptor[checksum]:
            raise ValueError('Artifact checksum mismatch: ' + descriptor[field])


class KerasAdapter:
    def is_portable(self, model, model_factory=None):
        return True

    def __init__(self, name='keras'):
        self.name = name

    def _module(self):
        if self.name == 'tensorflow':
            import tensorflow as tf
            return tf.keras
        return importlib.import_module('keras')

    def save(self, model, path, model_factory=None):
        target = str(Path(path).absolute()) + '.keras'
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        model.save(target)
        objects = {}
        for layer in [model, *getattr(model, 'layers', ())]:
            cls = type(layer)
            ref = reference(cls)
            if ref and not cls.__module__.startswith(('keras', 'tensorflow')):
                objects[cls.__name__] = ref
        descriptor = {'backend': self.name, 'format': 'keras', 'path': target, 'sha256': _digest(target)}
        if objects:
            descriptor['custom_objects'] = objects
        return descriptor

    def load(self, descriptor, custom_objects=None, model_factory=None):
        _verify(descriptor)
        objects = {name: resolve(value) for name, value in descriptor.get('custom_objects', {}).items()}
        objects.update(custom_objects or {})
        module = self._module()
        if descriptor.get('format') == 'keras_json_weights':
            config_text = Path(descriptor['path']).read_text()
            config = json.loads(config_text)
            if hasattr(module, 'ops') and 'module' not in config:
                # Let Keras' public legacy H5 loader translate Keras 2 configs.
                import h5py
                with tempfile.TemporaryDirectory(prefix='talos-legacy-model-') as temp:
                    combined = Path(temp) / 'legacy.h5'
                    with h5py.File(descriptor['weights'], 'r') as weights, h5py.File(combined, 'w') as output:
                        output.attrs['model_config'] = config_text
                        group = output.create_group('model_weights')
                        for key, value in weights.attrs.items():
                            group.attrs[key] = value
                        for key in weights:
                            weights.copy(key, group)
                    return module.models.load_model(combined, custom_objects=objects, compile=False)
            model = module.models.model_from_json(config_text, custom_objects=objects)
            model.load_weights(descriptor['weights'])
            return model
        return module.models.load_model(descriptor['path'], custom_objects=objects, compile=False)

    def predict(self, model, x, **kwargs):
        kwargs.setdefault('verbose', 0)
        return model.predict(x, **kwargs)

    def cleanup(self):
        self._module().backend.clear_session()
        gc.collect()


class TorchAdapter:
    name = 'torch'

    def _factory_spec(self, model, model_factory=None):
        factory = model_factory or getattr(model, 'talos_factory', None) or getattr(model, 'model_factory', None)
        config = getattr(model, 'talos_config', None)
        if config is None and hasattr(model, 'get_config'):
            config = model.get_config()
        if isinstance(factory, tuple):
            if len(factory) != 2 or not isinstance(factory[1], Mapping):
                raise TypeError('Torch factory tuples require (factory, configuration dictionary).')
            factory, config = factory
        if config is not None and not isinstance(config, Mapping):
            raise TypeError('Torch constructor configuration must be a dictionary.')
        if factory is None:
            cls = type(model)
            if cls.__module__ == 'torch.nn.modules.container' and cls.__name__ in ('Sequential', 'ModuleList', 'ModuleDict'):
                return None, dict(config or {})
            try:
                inspect.signature(cls).bind()
                factory = cls
            except (TypeError, ValueError):
                if config is not None:
                    factory = cls
        return factory, dict(config or {})

    def is_portable(self, model, model_factory=None):
        factory, _ = self._factory_spec(model, model_factory)
        return reference(factory) is not None

    def save(self, model, path, model_factory=None):
        import torch
        target = str(Path(path).absolute()) + '.pt'
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        # Copy before the next trial can change the live module or release a device.
        state = {key: value.detach().cpu().clone() if hasattr(value, 'detach') else copy.deepcopy(value)
                 for key, value in model.state_dict().items()}
        torch.save(state, target)
        factory, config = self._factory_spec(model, model_factory)
        factory_ref = reference(factory)
        resolved_factory = resolve(factory) if factory_ref else None
        source = inspect.getsourcefile(resolved_factory) if resolved_factory is not None and (inspect.isfunction(resolved_factory) or inspect.isclass(resolved_factory)) else None
        if source is not None and not Path(source).is_file():
            source = None
        return {'backend': 'torch', 'format': 'torch_state_dict', 'path': target, 'sha256': _digest(target),
                'factory': factory_ref, 'factory_source': source,
                'factory_source_sha256': _digest(source) if source else None, 'config': config or {},
                'factory_required': factory_ref is None}

    def load(self, descriptor, custom_objects=None, model_factory=None):
        _verify(descriptor)
        import torch
        factory = model_factory or descriptor.get('factory')
        config = descriptor.get('config', {})
        if isinstance(factory, tuple):
            if len(factory) != 2 or not isinstance(factory[1], Mapping):
                raise TypeError('Torch factory tuples require (factory, configuration dictionary).')
            factory, config = factory
        if factory is None:
            raise ValueError('Restoring a Torch state_dict requires an importable model_factory or an explicit factory.')
        factory = resolve(factory, descriptor.get('factory_source') if isinstance(factory, str) else None)
        model = factory(**config)
        try:
            state = torch.load(descriptor['path'], map_location='cpu', weights_only=True)
        except TypeError:  # Torch versions predating weights_only.
            state = torch.load(descriptor['path'], map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        return model

    def predict(self, model, x, **kwargs):
        import torch
        import numpy as np
        if hasattr(model, 'predict'):
            return model.predict(x, **kwargs)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        def tensor(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            return torch.as_tensor(value, dtype=torch.float32, device=device)
        training = model.training
        model.eval()
        with torch.no_grad():
            if isinstance(x, dict):
                result = model(**{key: tensor(value) for key, value in x.items()})
            elif isinstance(x, (list, tuple)):
                result = model(*[tensor(value) for value in x])
            else:
                result = model(tensor(x))
        model.train(training)
        def array(value):
            if isinstance(value, dict):
                return {key: array(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(array(item) for item in value)
            return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
        return array(result)

    def cleanup(self):
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GenericAdapter:
    name = 'generic'

    def is_portable(self, model, model_factory=None):
        return reference(type(model)) is not None

    def predict(self, model, x, **kwargs):
        return model.predict(x)

    def save(self, model, path, model_factory=None):
        import pickle
        target = str(Path(path).absolute()) + '.pkl'
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'wb') as stream:
            pickle.dump(model, stream)
        return {'backend': self.name, 'format': 'pickle', 'path': target, 'sha256': _digest(target)}

    def load(self, descriptor, custom_objects=None, model_factory=None):
        _verify(descriptor)
        import pickle
        with open(descriptor['path'], 'rb') as stream:
            return pickle.load(stream)

    def cleanup(self):
        gc.collect()


def backend_for(model=None, backend=None):
    aliases = {'tf': 'tensorflow', 'tf.keras': 'tensorflow', 'pytorch': 'torch'}
    if hasattr(backend, 'save') and hasattr(backend, 'predict'):
        return backend
    name = aliases.get(backend, backend)
    if name is None and model is not None:
        modules = [cls.__module__ for cls in type(model).__mro__]
        if any(module.startswith('keras') for module in modules):
            name = 'keras'
        elif any(module.startswith('tensorflow') for module in modules):
            name = 'tensorflow'
        elif any(module.startswith('torch') for module in modules):
            name = 'torch'
    if name in ('keras', 'tensorflow'):
        return KerasAdapter(name)
    if name == 'torch':
        return TorchAdapter()
    if name in (None, 'generic'):
        return GenericAdapter()
    raise ValueError('Unsupported backend: ' + str(name))
