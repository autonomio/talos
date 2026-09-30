import hashlib
import importlib.metadata
import inspect
import marshal
import platform
import subprocess
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    """Identify supplied data without acquiring or consuming a dataset."""
    if value is None:
        return {'kind': 'none'}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {'kind': 'array', 'shape': list(value.shape), 'values': fingerprint(value.tolist())}
        array = np.ascontiguousarray(value)
        return {'kind': 'array', 'shape': list(array.shape), 'dtype': str(array.dtype),
                'sha256': hashlib.sha256(array.view(np.uint8)).hexdigest()}
    if isinstance(value, dict):
        return {str(key): fingerprint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [fingerprint(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    module = type(value).__module__.split('.')[0]
    if module in ('pandas', 'polars'):
        return {'kind': module, 'columns': list(value.columns), 'values': fingerprint(value.to_numpy())}
    if module == 'torch' or any(cls.__module__.startswith('torch') for cls in type(value).__mro__):
        if hasattr(value, 'detach'):
            import torch
            tensor = value.detach().cpu()
            if tensor.layout == torch.strided and not tensor.is_quantized:
                tensor = tensor.resolve_conj().resolve_neg().contiguous()
                raw = tensor.reshape(-1).view(torch.uint8).numpy()
                return {'kind': 'torch', 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                        'sha256': hashlib.sha256(raw.tobytes()).hexdigest()}
            return {'kind': 'opaque', 'type': f'{type(value).__module__}.{type(value).__qualname__}',
                    'fingerprint_required': True}
    if module in ('tensorflow', 'keras') and hasattr(value, 'numpy'):
        return fingerprint(value.numpy())
    return {'kind': 'opaque', 'type': f'{type(value).__module__}.{type(value).__qualname__}',
            'fingerprint_required': True}


def source_identity(function):
    if function is None:
        return None
    identity = {'module': getattr(function, '__module__', type(function).__module__),
                'qualname': getattr(function, '__qualname__', type(function).__qualname__)}
    target = function.__func__ if inspect.ismethod(function) else function
    path = inspect.getsourcefile(target) if inspect.isfunction(target) or inspect.isclass(target) else None
    if path and Path(path).is_file():
        identity['source_path'] = str(Path(path).resolve())
        identity['source_sha256'] = file_hash(path)
        revision = subprocess.run(['git', '-C', str(Path(path).parent), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        if revision.returncode == 0:
            identity['git_revision'] = revision.stdout.strip()
    code = getattr(target, '__code__', None)
    if code is not None:
        identity['bytecode_sha256'] = hashlib.sha256(marshal.dumps(code)).hexdigest()
    return identity


def environment():
    names = ('talos', 'numpy', 'pandas', 'polars', 'scikit-learn', 'tensorflow', 'tensorflow-macos', 'keras', 'torch')
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    from talos import __version__
    packages['talos'] = __version__
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}
