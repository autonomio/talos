from contextlib import contextmanager
from importlib import import_module, util
from pathlib import Path
import hashlib
import sys


def load_sfd(reference, base_path=None):
    """Resolve caller SFD code; loading input observations remains caller code."""
    source = Path(reference)
    if source.suffix == '.py':
        if not source.is_absolute():
            base = Path(base_path or Path.cwd())
            candidates = [base / source]
            from talos.yaml.config import find_project_root
            project = find_project_root(base)
            if project is not None:
                candidates.extend([project / source, project / 'manifests' / source])
            source = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f'SFD source not found: {source}')
        name = '_talos_sfd_' + hashlib.sha256(str(source).encode()).hexdigest()[:16]
        spec = util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load SFD source: {source}')
        module = util.module_from_spec(spec)
        sys.modules[name] = module
        sys.path.insert(0, str(source.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
    else:
        module = import_module(reference)
    return validate_sfd(module, reference)


def validate_sfd(module, reference='caller'):
    if not callable(getattr(module, 'params', None)):
        raise TypeError(f'SFD {reference} must expose callable params')
    if all(callable(getattr(module, name, None)) for name in ('prep', 'model')):
        return module
    manifest = getattr(module, 'manifest', None)
    if callable(manifest) or (manifest is not None and all(
            callable(getattr(manifest, name, None)) for name in ('prepare_data', 'run_model'))):
        return module
    raise TypeError(f'SFD {reference} must expose prep/model or a manifest factory/object')


def resolve(reference):
    """Resolve an explicit importable callable reference, never a literal string."""
    module_name, separator, attribute = reference.partition(':')
    if not separator:
        module_name, _, attribute = reference.rpartition('.')
    if not module_name or not attribute:
        raise ValueError(f'Expected module:callable reference, got {reference!r}')
    value = import_module(module_name)
    for name in attribute.split('.'):
        value = getattr(value, name)
    if not callable(value):
        raise TypeError(f'{reference} is not callable')
    return value


def resolve_values(value):
    if isinstance(value, dict):
        if set(value) == {'callable'}:
            return resolve(value['callable'])
        return {key: resolve_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_values(item) for item in value)
    return value


@contextmanager
def caller_imports(module):
    """Resolve caller-local imports while invoking caller code or manifest references."""
    source = getattr(module, '__file__', None)
    paths = []
    if source:
        paths.append(str(Path(source).resolve().parent))
        from talos.yaml.config import find_project_root
        project = find_project_root(Path(source).resolve().parent)
        if project is not None and str(project) not in paths:
            paths.append(str(project))
    sys.path[:0] = paths
    try:
        yield
    finally:
        for path in paths:
            sys.path.remove(path)
