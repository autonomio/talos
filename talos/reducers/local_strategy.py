"""Audit and recover live model, data and local-file strategy changes."""

import hashlib
import sys
import types
from pathlib import Path

from talos.experiment.serialization import callable_reference, content_hash, decode, encode

_CONTROL_FIELDS = ('model', 'reduction_method', 'reduction_interval', 'reduction_window',
                   'reduction_threshold', 'reduction_metric', 'minimize_loss', 'performance_target',
                   'print_params', 'save_models', 'save_weights', 'clear_session')
_DATA_FIELDS = ('x_train', 'y_train', 'x_val', 'y_val')


def _dense_tensor(value):
    """Return a dense tensor's portable metadata and NumPy payload lazily."""
    import numpy as np
    modules = [cls.__module__ for cls in type(value).__mro__]
    if any(module.startswith('torch') for module in modules) and hasattr(value, 'detach'):
        import torch
        if value.layout != torch.strided or value.is_quantized:
            return None
        if type(value) not in (torch.Tensor, torch.nn.Parameter):
            return None
        array = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            array = array.float()
        return {'kind': 'torch', 'dtype': str(value.dtype).split('.')[-1],
                'class': type(value).__name__, 'device': str(value.device),
                'requires_grad': bool(value.requires_grad)}, array.numpy()
    if any(module.startswith(('tensorflow', 'keras')) for module in modules) and hasattr(value, 'numpy'):
        import tensorflow as tf
        if not tf.is_tensor(value) and not isinstance(value, tf.Variable):
            return None
        array = value.numpy()
        if value.dtype.name == 'bfloat16':
            array = np.asarray(array, dtype=np.float32)
        return {'kind': 'tensorflow', 'dtype': value.dtype.name,
                'class': type(value).__name__, 'variable': isinstance(value, tf.Variable),
                'trainable': bool(getattr(value, 'trainable', False))}, array
    return None


def _data_fingerprint(value):
    from talos.experiment.provenance import fingerprint
    if isinstance(value, dict):
        return {str(key): _data_fingerprint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_data_fingerprint(item) for item in value]
    tensor = _dense_tensor(value)
    if tensor is not None:
        metadata, array = tensor
        return {**metadata, 'payload': fingerprint(array)}
    result = fingerprint(value)
    if isinstance(result, dict) and result.get('kind') == 'opaque':
        # Identity only detects opaque replacement; no stream is consumed.
        result['instance_id'] = id(value)
    return result


def _snapshot_data(value):
    import numpy as np
    if isinstance(value, dict):
        return {'kind': 'mapping', 'items': [[encode(key), _snapshot_data(item)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return {'kind': 'tuple' if isinstance(value, tuple) else 'list',
                'items': [_snapshot_data(item) for item in value]}
    tensor = _dense_tensor(value)
    if tensor is not None:
        metadata, array = tensor
        return {**metadata, 'payload': encode(array)}
    if value is None or isinstance(value, (np.ndarray, np.generic, str, bytes, bool, int, float, complex)):
        return {'kind': 'value', 'value': encode(value)}
    return {'kind': 'nonportable', 'type': f'{type(value).__module__}.{type(value).__qualname__}'}


def _restore_data(state):
    kind = state['kind']
    if kind == 'mapping':
        return {decode(key): _restore_data(item) for key, item in state['items']}
    if kind in ('list', 'tuple'):
        values = [_restore_data(item) for item in state['items']]
        return tuple(values) if kind == 'tuple' else values
    if kind == 'value':
        return decode(state['value'])
    if kind == 'torch':
        import torch
        tensor = torch.tensor(decode(state['payload']), dtype=getattr(torch, state['dtype']),
                              device=state['device'])
        if state['class'] == 'Parameter':
            return torch.nn.Parameter(tensor, requires_grad=state['requires_grad'])
        return tensor.requires_grad_(state['requires_grad'])
    if kind == 'tensorflow':
        import tensorflow as tf
        dtype = tf.as_dtype(state['dtype'])
        if state['variable']:
            return tf.Variable(decode(state['payload']), dtype=dtype, trainable=state['trainable'])
        return tf.convert_to_tensor(decode(state['payload']), dtype=dtype)
    from talos.experiment.errors import NonPortableValueError
    raise NonPortableValueError(f"Cannot resume changed nonportable live data {state.get('type', kind)}; supply dense arrays or tensors.")


def _controls(scan):
    controls = {name: encode(getattr(scan, name)) for name in _CONTROL_FIELDS if hasattr(scan, name)}
    controls.update({name: _data_fingerprint(getattr(scan, name)) for name in _DATA_FIELDS if hasattr(scan, name)})
    return controls


def capture_live_controls(scan, source_model):
    """Capture changed live controls without serializing unchanged caller data."""
    baseline = getattr(scan, '_local_controls_baseline', None)
    if baseline is None:
        return None
    current = _controls(scan)
    from talos.experiment.provenance import source_identity
    model_changed = (source_identity(scan.model) != source_identity(source_model) or
                     content_hash(encode(scan.model)) != content_hash(encode(source_model)))
    controls = {name: value for name, value in current.items() if name in _CONTROL_FIELDS and name != 'model' and
                content_hash(value) != content_hash(baseline.get(name))}
    data = {name: _snapshot_data(getattr(scan, name)) for name in _DATA_FIELDS if name in current and
            content_hash(current[name]) != content_hash(baseline.get(name))}
    if not controls and not data and not model_changed:
        return None
    from talos.experiment.source_snapshot import snapshot_sources
    run_dir = Path(scan._experiment_log).parent
    strategy = getattr(scan, '_local_strategy_function', None)
    model = scan.model if model_changed else None
    bundle = snapshot_sources(model, strategy, [getattr(scan, name) for name in controls], run_dir)
    return {'controls': controls, 'data': data,
            'model_reference': callable_reference(model) if model_changed else None,
            'model_sources': bundle,
            'strategy_reference': callable_reference(strategy) if strategy is not None else None,
            'strategy_source_hash': getattr(scan, '_local_strategy_hash', None)}


def restore_live_controls(scan, state):
    """Restore saved controls after code/configuration identity validation."""
    if state is None:
        return
    from talos.experiment.source_snapshot import hydrate_sources, verify_sources
    run_dir = Path(scan._experiment_log).parent
    bundle = state.get('model_sources')
    if bundle:
        verify_sources(bundle, run_dir, originals=True)
        hydrate_sources(bundle, run_dir)
    restored = {}
    for name, value in state.get('controls', {}).items():
        if name not in _CONTROL_FIELDS or name == 'model':
            raise ValueError(f'Unknown persisted live control: {name}')
        restored[name] = decode(value)
    for name, value in state.get('data', {}).items():
        if name not in _DATA_FIELDS:
            raise ValueError(f'Unknown persisted live data: {name}')
        restored[name] = _restore_data(value)
    if state.get('model_reference') is not None:
        restored['model'] = decode(state['model_reference'])
    if state.get('strategy_reference') is not None:
        restored['_local_strategy_function'] = decode(state['strategy_reference'])
        restored['_local_strategy_hash'] = state['strategy_source_hash']
    scan._local_controls_baseline = _controls(scan)
    for name, value in restored.items():
        setattr(scan, name, value)


def _source_snapshot(scan, source, digest):
    if not getattr(scan, '_experiment_log', None):
        return None
    run_dir = Path(scan._experiment_log).parent
    snapshot = run_dir / 'sources' / 'local_strategy' / (digest + '.py')
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError('Local strategy source snapshot checksum mismatch.')
    else:
        snapshot.write_bytes(source)
    return snapshot


def local_strategy(self):
    path = Path.cwd() / 'talos_strategy.py'
    if not path.exists():
        raise FileNotFoundError(f'Local strategy file does not exist: {path}')
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    previous_digest = getattr(self, '_local_strategy_hash', None)
    snapshot = _source_snapshot(self, source, digest)
    if previous_digest != digest:
        name = '_talos_local_strategy_' + digest
        module = types.ModuleType(name)
        module.__file__ = str(snapshot or path)
        sys.modules[name] = module
        exec(compile(source, str(snapshot or path), 'exec'), module.__dict__)
        function = getattr(module, 'talos_strategy', None)
        if not callable(function):
            raise TypeError('talos_strategy.py must define callable talos_strategy(scan).')
        self._local_strategy_function = function
        self._local_strategy_hash = digest
    queue = getattr(getattr(self, 'param_object', None), '_msq', None)
    if queue is not None and previous_digest != digest:
        snapshot_path = str(snapshot.relative_to(Path(self._experiment_log).parent)) if snapshot else None
        queue._log_intervention('legacy_local_source_revision', source='local_strategy',
            source_hash=digest, previous_source_hash=previous_digest,
            source_path=str(path), snapshot_path=snapshot_path)
    before = _controls(self)
    if not hasattr(self, '_local_controls_baseline'):
        self._local_controls_baseline = before
    result = self._local_strategy_function(self)
    scan = self if result is None else result
    after = _controls(scan)
    changed = {name: {'before': before.get(name), 'after': after.get(name)}
               for name in before.keys() | after.keys()
               if content_hash(before.get(name)) != content_hash(after.get(name))}
    if queue is not None and changed:
        queue._log_intervention('legacy_local_control_change', source='local_strategy',
                               source_hash=digest, changes=changed)
    return scan
