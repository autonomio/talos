"""Locate project roots and configure the experiment manifest store."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from typing import Any, Protocol, TypeGuard

from ruamel.yaml import YAML

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


_TALOS_TOML_NAME = 'talos.toml'
STORE_RELATIVE = Path('manifests') / 'committed'


class RoundTripYAML(Protocol):

    '''Typed facade over a ruamel round-trip YAML instance, mirroring its declared load/dump contract.'''

    def load(self, stream: str) -> Any: ...

    def dump(self, data: dict[str, Any], stream: StringIO) -> None: ...


def round_trip_yaml(preserve_quotes: bool = False, width: int | None = None) -> RoundTripYAML:

    '''
    Create a ruamel round-trip YAML instance behind the typed facade.

    Args:
        preserve_quotes (bool): Preserve quoting styles on round-trip when True
        width (int | None): Dump line width, or None for the ruamel default

    Returns:
        RoundTripYAML: Configured round-trip YAML instance

    '''

    yaml = YAML()
    if preserve_quotes:
        yaml.preserve_quotes = True
    if width is not None:
        yaml.width = width
    return yaml


def is_mapping(value: Any) -> TypeGuard[dict[str, Any]]:

    '''
    Check whether a parsed YAML value is a mapping.

    Args:
        value (Any): Candidate value from a parsed YAML document

    Returns:
        TypeGuard[dict[str, Any]]: True if value is a dict

    '''

    return isinstance(value, dict)


def is_list(value: Any) -> TypeGuard[list[Any]]:

    '''
    Check whether a parsed YAML value is a list.

    Args:
        value (Any): Candidate value from a parsed YAML document

    Returns:
        TypeGuard[list[Any]]: True if value is a list

    '''

    return isinstance(value, list)


def find_project_root(start: Path) -> Path | None:

    '''
    Walk up the directory tree looking for talos.toml.

    Args:
        start (Path): Directory to start searching from

    Returns:
        Path | None: The directory containing talos.toml, or None if not found

    '''

    current = start.resolve()
    while True:
        if (current / _TALOS_TOML_NAME).exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def read_talos_toml(project_root: Path) -> dict[str, Any]:

    '''
    Read talos.toml from the project root.

    Args:
        project_root (Path): Directory containing talos.toml

    Returns:
        dict: Parsed contents of talos.toml

    '''

    toml_path = project_root / _TALOS_TOML_NAME
    if not toml_path.exists():
        raise FileNotFoundError(f"talos.toml not found at '{toml_path}'")

    return tomllib.loads(toml_path.read_text(encoding='utf-8'))


def get_store_path(start: Path) -> Path:

    '''
    Locate the manifest store by finding the project root.

    Args:
        start (Path): Directory to start searching from

    Returns:
        Path: Path to manifests/committed/

    Raises:
        FileNotFoundError: If no talos.toml is found in the directory tree

    '''

    root = find_project_root(start)
    if root is None:
        raise FileNotFoundError(
            'No talos.toml found. Run talos new <project-name> to create a project.'
        )
    return root / STORE_RELATIVE
