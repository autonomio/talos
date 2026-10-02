"""Canonical scientific records; live Python objects never become repr strings."""
import base64
import hashlib
import importlib
import inspect
import json
import marshal
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .errors import NonPortableValueError

_TAG = '__talos_type__'
__all__ = ['callable_reference', 'content_hash', 'decode', 'dumps', 'encode']


def _resolve(module, qualname):
    value = importlib.import_module(module)
    for part in qualname.split('.'):
        value = getattr(value, part)
    return value


def callable_reference(func, *, _seen=None):
    if func is None:
        return None
    module = getattr(func, '__module__', None)
    name = getattr(func, '__name__', None)
    qualname = getattr(func, '__qualname__', name)
    if module is None and type(func).__module__.split('.')[0] == 'numpy':
        module = 'numpy'
    if module and module != '__main__' and name:
        try:
            exported = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            exported = None
        if exported is func:
            qualname = name

    if module and module != '__main__' and qualname and '<' not in qualname:
        try:
            resolved = _resolve(module, qualname)
        except (ImportError, AttributeError):
            resolved = None
        if resolved is func or (inspect.ismethod(resolved) and inspect.ismethod(func) and
                                resolved.__func__ is func.__func__ and resolved.__self__ is func.__self__):
            reference = {_TAG: 'callable', 'module': module, 'qualname': qualname, 'portable': True}
            try:
                source = inspect.getsource(func).encode()
            except (OSError, TypeError):
                source = None
            if source is not None:
                reference['source_hash'] = hashlib.sha256(source).hexdigest()
            return reference
    try:
        source = inspect.getsource(func).encode()
    except (OSError, TypeError):
        code = getattr(func, '__code__', None)
        source = marshal.dumps(code) if code is not None else type(func).__qualname__.encode()
    reference = {_TAG: 'nonportable_callable', 'module': module, 'qualname': qualname,
                 'source_hash': hashlib.sha256(source).hexdigest(), 'portable': False}
    seen = set() if _seen is None else _seen
    if id(func) in seen:
        reference['recursive'] = True
        return reference
    seen.add(id(func))
    try:
        defaults = getattr(func, '__defaults__', None)
        if defaults:
            reference['defaults'] = _capture(defaults, seen)
        keyword_defaults = getattr(func, '__kwdefaults__', None)
        if keyword_defaults:
            reference['keyword_defaults'] = _capture(keyword_defaults, seen)
        closure = getattr(func, '__closure__', None)
        if closure:
            reference['closure'] = [_capture(cell.cell_contents, seen) for cell in closure]
        for name in ('func', 'args', 'keywords', '__wrapped__'):
            nested = getattr(func, name, None)
            if nested is not None:
                reference[name] = _capture(nested, seen)
        owner = getattr(func, '__self__', None)
        if owner is not None:
            reference['bound_owner'] = _capture(owner if isinstance(owner, type) else
                                                {'type': type(owner), 'state': getattr(owner, '__dict__', None)}, seen)
        if not inspect.isfunction(func) and not inspect.isclass(func):
            state = getattr(func, '__dict__', None)
            if state:
                reference['state'] = _capture(state, seen)
    finally:
        seen.remove(id(func))
    return reference


def _capture(value, seen):
    try:
        return encode(value, _seen=seen)
    except (NonPortableValueError, RecursionError):
        return {_TAG: 'nonportable_state', 'module': type(value).__module__,
                'qualname': type(value).__qualname__, 'portable': False}


def encode(value, *, _seen=None):
    if type(value).__module__.startswith('numpy'):
        import numpy as np
        if isinstance(value, np.ndarray):
            return {_TAG: 'ndarray', 'dtype': _dtype_descriptor(value.dtype), 'shape': list(value.shape), 'items': encode(list(value.flat), _seen=_seen)}
        if isinstance(value, np.generic):
            native = value.item()
            if isinstance(native, np.generic):
                return {_TAG: 'numpy_raw_scalar', 'dtype': _dtype_descriptor(value.dtype), 'scalar_class': type(value).__name__, 'bytes': value.tobytes().hex()}
            return {_TAG: 'numpy_scalar', 'dtype': _dtype_descriptor(value.dtype), 'scalar_class': type(value).__name__, 'value': encode(native, _seen=_seen)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {_TAG: 'float', 'value': 'nan' if math.isnan(value) else ('inf' if value > 0 else '-inf')}
    if isinstance(value, complex):
        return {_TAG: 'complex', 'real': encode(value.real), 'imag': encode(value.imag)}
    if isinstance(value, bytes):
        return {_TAG: 'bytes', 'value': base64.b64encode(value).decode('ascii')}
    if isinstance(value, (datetime, date, time)):
        return {_TAG: type(value).__name__, 'value': value.isoformat()}
    if isinstance(value, timedelta):
        return {_TAG: 'timedelta', 'days': value.days, 'seconds': value.seconds, 'microseconds': value.microseconds}
    if callable(value):
        return callable_reference(value, _seen=_seen)
    if isinstance(value, Path):
        return {_TAG: 'path', 'value': str(value)}
    if isinstance(value, tuple):
        return {_TAG: 'tuple', 'items': [encode(item, _seen=_seen) for item in value]}
    if isinstance(value, list):
        return [encode(item, _seen=_seen) for item in value]
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value) and _TAG not in value:
            return {key: encode(item, _seen=_seen) for key, item in value.items()}
        items = [[encode(key, _seen=_seen), encode(item, _seen=_seen)] for key, item in value.items()]
        items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
        return {_TAG: 'mapping', 'items': items}
    if isinstance(value, (set, frozenset)):
        items = sorted((encode(item, _seen=_seen) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
        return {_TAG: 'set' if isinstance(value, set) else 'frozenset', 'items': items}
    raise NonPortableValueError(f'Cannot serialize {type(value).__module__}.{type(value).__qualname__}; use an importable factory or plain values.')


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get(_TAG)
    if kind is None:
        return {key: decode(item) for key, item in value.items()}
    if kind == 'callable':
        try:
            function = _resolve(value['module'], value['qualname'])
            expected = value.get('source_hash')
            if expected is not None and callable_reference(function).get('source_hash') != expected:
                raise NonPortableValueError(f"Persisted callable source changed: {value['module']}.{value['qualname']}")
            return function
        except (ImportError, AttributeError) as error:
            raise NonPortableValueError(f"Cannot import persisted callable {value['module']}.{value['qualname']}") from error
    if kind == 'nonportable_state':
        raise NonPortableValueError(f"Cannot resume captured nonportable state {value['module']}.{value['qualname']}.")
    if kind == 'nonportable_callable':
        raise NonPortableValueError(f"Cannot resume nonportable callable {value.get('qualname')}; define it in an importable module.")
    if kind == 'float':
        return float(value['value'])
    if kind == 'complex':
        return complex(decode(value['real']), decode(value['imag']))
    if kind == 'bytes':
        return base64.b64decode(value['value'])
    if kind in ('date', 'datetime', 'time'):
        return {'date': date, 'datetime': datetime, 'time': time}[kind].fromisoformat(value['value'])
    if kind == 'timedelta':
        return timedelta(days=value['days'], seconds=value['seconds'], microseconds=value['microseconds'])
    if kind == 'path':
        return Path(value['value'])
    if kind == 'tuple':
        return tuple(decode(item) for item in value['items'])
    if kind == 'mapping':
        return {decode(key): decode(item) for key, item in value['items']}
    if kind in ('set', 'frozenset'):
        factory = set if kind == 'set' else frozenset
        return factory(decode(item) for item in value['items'])
    if kind in ('ndarray', 'numpy_scalar', 'numpy_raw_scalar'):
        import numpy as np
        dtype = _dtype_restore(value['dtype'])
        if kind == 'numpy_raw_scalar':
            return _numpy_scalar_type(np.frombuffer(bytes.fromhex(value['bytes']), dtype=dtype)[0], value.get('scalar_class'))
        if kind == 'ndarray':
            items = decode(value['items'])
            array = np.empty(value['shape'], dtype=dtype)
            for index, item in enumerate(items):
                array.flat[index] = item
            return array
        return _numpy_scalar_type(np.array(decode(value['value']), dtype=dtype)[()], value.get('scalar_class'))
    raise ValueError(f'Unknown serialized Talos value type: {kind}')


def _numpy_scalar_type(value, class_name):
    if class_name is None or type(value).__name__ == class_name:
        return value
    import numpy as np
    scalar_class = getattr(np, class_name, None)
    if not isinstance(scalar_class, type) or not issubclass(scalar_class, np.generic):
        raise ValueError(f'Unknown NumPy scalar class: {class_name}')
    return scalar_class(value)


def _dtype_descriptor(dtype):
    return {'fields': encode(dtype.descr)} if dtype.fields else str(dtype)


def _dtype_restore(value):
    import numpy as np
    return np.dtype(decode(value['fields'])) if isinstance(value, dict) else np.dtype(value)


def dumps(value):
    return json.dumps(encode(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def content_hash(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()
