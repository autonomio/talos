"""Verified snapshots of caller Python modules, excluding installed dependencies."""
import ast
import hashlib
import importlib
import importlib.util
import importlib.machinery
import importlib.metadata
import inspect
import sys
import sysconfig
from pathlib import Path
from types import ModuleType

_DIRECTORY = 'sources/modules'


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _distribution_paths():
    """Distribution-owned modules also cover pip --target and alternate sys.path roots."""
    directories, files = set(), set()
    for distribution in importlib.metadata.distributions():
        top_levels = set((distribution.read_text('top_level.txt') or '').split())
        for entry in distribution.files or ():
            parts = entry.parts
            if parts and parts[0] not in ('.', '..'):
                top_levels.add(parts[0])
        for name in top_levels:
            if name.endswith(('.dist-info', '.egg-info')):
                continue
            path = Path(distribution.locate_file(name)).resolve()
            if path.is_dir():
                directories.add(path)
            elif path.is_file() and path.suffix in ('.py', '.pyc', '.so', '.pyd'):
                files.add(path)
    return directories, files


def _module_spec(name, roots):
    """Resolve exact package paths without importing optional dependency integrations."""
    paths = [str(root) for root in sorted(roots, key=str)] + list(sys.path)
    parts = name.split('.')
    for count in range(1, len(parts) + 1):
        specification = importlib.machinery.PathFinder.find_spec('.'.join(parts[:count]), paths)
        if specification is None:
            return None
        if count < len(parts):
            if specification.submodule_search_locations is None:
                return None
            paths = list(specification.submodule_search_locations)
    return specification


def _local_path(path, distributions=None):
    if not path:
        return None
    path = Path(path).resolve()
    if not path.exists():
        return None
    if distributions is not None:
        directories, files = distributions
        if path in files or any(path.is_relative_to(directory) for directory in directories):
            return None
    if {'site-packages', 'dist-packages'} & set(path.parts):
        return None
    for name in ('stdlib', 'purelib', 'platlib'):
        prefix = sysconfig.get_path(name)
        if prefix and path.is_relative_to(Path(prefix).resolve()):
            return None
    return path


def _local_source(path, distributions=None):
    path = _local_path(path, distributions)
    return path if path is not None and path.suffix == '.py' and path.is_file() else None


def _valid_name(name):
    return isinstance(name, str) and name != '__main__' and all(part.isidentifier() for part in name.split('.'))


def _walk_callables(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if callable(value):
        yield value
        for attribute in ('func', '__wrapped__'):
            nested = getattr(value, attribute, None)
            if nested is not None:
                yield from _walk_callables(nested, seen)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_callables(item, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _walk_callables(item, seen)
    elif type(value).__module__.startswith('numpy') and getattr(value, 'dtype', None) is not None and value.dtype.hasobject:
        for item in value.flat:
            yield from _walk_callables(item, seen)


def snapshot_sources(model_func, prep_func, params, run_dir):
    """Snapshot caller roots, local imports and importable callable candidate modules."""
    run_dir = Path(run_dir).resolve()
    pending = []
    roots = set()
    discovered = {}
    entries = []
    distributions = _distribution_paths()

    def add(name, path=None, package=None):
        if not _valid_name(name) or name == 'talos' or name.startswith('talos.') or name in discovered:
            return
        module = sys.modules.get(name)
        known_path = getattr(module, '__file__', None)
        namespace = None
        if path is None:
            if module is not None:
                path = _local_source(known_path, distributions)
                if path is None and getattr(module, '__path__', None) and known_path is None:
                    namespace = next((local for candidate in module.__path__
                                      if (local := _local_path(candidate, distributions)) is not None), None)
                    package = True
                if path is None and namespace is None:
                    return
            else:
                specification = _module_spec(name, roots)
                if specification is not None:
                    path = _local_source(specification.origin, distributions)
                    package = specification.submodule_search_locations is not None
                    if path is None and specification.origin is None and package:
                        namespace = next((local for candidate in specification.submodule_search_locations
                                          if (local := _local_path(candidate, distributions)) is not None), None)
        path = _local_source(path, distributions)
        if path is None and namespace is None:
            return
        package = path.name == '__init__.py' if package is None else package
        discovered[name] = {'source': path, 'namespace': namespace, 'package': bool(package)}
        import_root = path.parent if path else namespace
        for _ in range(len(name.split('.')) if package else len(name.split('.')) - 1):
            import_root = import_root.parent
        roots.add(import_root)
        pending.append(name)
        for count in range(1, len(name.split('.'))):
            add('.'.join(name.split('.')[:count]))

    for function in _walk_callables([model_func, prep_func, params]):
        name = getattr(function, '__module__', None)
        add(name)
        if name in discovered and name not in entries:
            entries.append(name)

    while pending:
        name = pending.pop(0)
        source = discovered[name]['source']
        module = sys.modules.get(name)
        if module is not None:
            for value in vars(module).values():
                dependency = value.__name__ if isinstance(value, ModuleType) else getattr(value, '__module__', None) if callable(value) else None
                # Import machinery attaches every loaded child to its package. These
                # incidental children are not dependencies of an empty parent file.
                if (isinstance(value, ModuleType) and discovered[name]['package']
                        and name not in entries and dependency.startswith(name + '.')):
                    continue
                add(dependency)
        if source is None:
            continue
        tree = ast.parse(source.read_bytes(), filename=str(source))
        package_name = name if discovered[name]['package'] else name.rpartition('.')[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imported = node.module or ''
                if node.level:
                    if not package_name:
                        continue
                    imported = importlib.util.resolve_name('.' * node.level + imported, package_name)
                add(imported)
                for alias in node.names:
                    if alias.name != '*':
                        add(imported + '.' + alias.name if imported else alias.name)

    modules = {}
    for name, specification in sorted(discovered.items()):
        parts = name.split('.')
        relative = Path(_DIRECTORY).joinpath(*parts)
        relative = relative / '__init__.py' if specification['package'] and specification['source'] else relative if specification['namespace'] else relative.with_suffix('.py')
        destination = run_dir / relative
        if specification['namespace'] is not None:
            destination.mkdir(parents=True, exist_ok=True)
            modules[name] = {'path': relative.as_posix(), 'sha256': None, 'namespace': True,
                             'original_path': str(specification['namespace']), 'package': True}
            continue
        content = specification['source'].read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if _digest(destination) != digest:
                raise ValueError(f'Caller source changed since its snapshot: {name}')
        else:
            destination.write_bytes(content)
        modules[name] = {'path': relative.as_posix(), 'sha256': digest,
                         'original_path': str(specification['source']), 'package': specification['package']}
    return {'version': 1, 'directory': _DIRECTORY, 'modules': modules, 'entry_modules': entries}


def verify_sources(metadata, run_dir, *, originals=False):
    """Verify bundle files, optionally rejecting changed surviving original sources."""
    bundle = metadata.get('source_bundle', metadata) if isinstance(metadata, dict) else None
    if not bundle or 'modules' not in bundle:
        return
    if bundle.get('version') != 1:
        raise ValueError('Unsupported caller source snapshot version')
    run_dir = Path(run_dir).resolve()
    directory = (run_dir / bundle['directory']).resolve()
    if not directory.is_relative_to(run_dir):
        raise ValueError('Caller source snapshot directory lies outside run directory')
    modules = bundle['modules']
    for name, specification in modules.items():
        if not _valid_name(name):
            raise ValueError(f'Invalid saved module name: {name}')
        path = (run_dir / specification['path']).resolve()
        exists = path.is_dir() if specification.get('namespace') else path.is_file()
        if not path.is_relative_to(directory) or not exists:
            raise ValueError(f'Missing caller source snapshot: {name}')
        if not specification.get('namespace') and _digest(path) != specification['sha256']:
            raise ValueError(f'Caller source snapshot checksum mismatch: {name}')
        if originals and not specification.get('namespace'):
            original = Path(specification.get('original_path', ''))
            if original.is_file() and _digest(original) != specification['sha256']:
                raise ValueError(f'Caller source changed since its snapshot: {name}')
    return bundle


def hydrate_sources(metadata, run_dir):
    """Verify and load exact saved caller modules before decoding callable references."""
    bundle = verify_sources(metadata, run_dir)
    if not bundle:
        return
    run_dir = Path(run_dir).resolve()
    directory = (run_dir / bundle['directory']).resolve()
    modules = bundle['modules']
    if not modules:
        return
    # Retain identical cached modules: callers may still hold their callable identities.
    # Replace changed modules and bundled dependents, never unrelated package children.
    replaced = set()
    dependencies = {}
    for name, specification in modules.items():
        cached = sys.modules.get(name)
        if cached is not None and not specification.get('namespace'):
            cached_path = getattr(cached, '__file__', None)
            if not cached_path or not Path(cached_path).is_file() or _digest(cached_path) != specification['sha256']:
                replaced.add(name)
        imported = set()
        if cached is not None:
            for value in vars(cached).values():
                dependency = value.__name__ if isinstance(value, ModuleType) else getattr(value, '__module__', None) if callable(value) else None
                if dependency in modules and dependency != name:
                    imported.add(dependency)
        if not specification.get('namespace'):
            tree = ast.parse((run_dir / specification['path']).read_bytes())
            package = name if specification['package'] else name.rpartition('.')[0]
            for node in ast.walk(tree):
                candidates = []
                if isinstance(node, ast.Import):
                    candidates = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    reference = node.module or ''
                    if node.level:
                        if not package:
                            continue
                        reference = importlib.util.resolve_name('.' * node.level + reference, package)
                    candidates = [reference, *(reference + '.' + alias.name for alias in node.names if alias.name != '*')]
                for candidate in candidates:
                    if candidate in modules and candidate != name:
                        imported.add(candidate)
        dependencies[name] = imported
    while True:
        affected = {name for name, imported in dependencies.items() if imported & replaced}
        if affected <= replaced:
            break
        replaced.update(affected)
    for name in replaced:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    try:
        for name in sorted(modules, key=lambda value: (value.count('.'), value)):
            module = importlib.import_module(name)
            if modules[name]['package']:
                saved_package = run_dir / modules[name]['path']
                if not modules[name].get('namespace'):
                    saved_package = saved_package.parent
                paths = list(module.__path__)
                if str(saved_package) not in paths:
                    module.__path__ = [str(saved_package), *paths]
    finally:
        sys.path.remove(str(directory))
