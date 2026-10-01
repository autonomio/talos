from pathlib import Path
from typing import Any

import click

from talos.yaml.config import find_project_root
from talos.yaml.config import is_mapping
from talos.yaml.store import SHA256_PREFIX
from talos.yaml.store import load_index
from talos.yaml.store import normalize_manifest_ref
from talos.yaml.store import resolve_manifest_uri
from talos.yaml.store import short_id


def run_lineage(ref: str, start: Path) -> bool:

    '''
    Display the lineage chain for a committed manifest.

    Walks the parent_id chain from the target up to the root and prints it
    root-first, with the target marked.

    Args:
        ref (str): Manifest reference (bare hash, sha256:<hash>, or manifest:// URI)
        start (Path): Directory to start searching for the project root

    Returns:
        bool: True on success, False on failure

    '''

    project_root = find_project_root(start)
    if project_root is None:
        click.secho('  ✗ No talos project found. Run this command from inside a Talos project.', fg='red')
        return False

    try:
        candidate, _ = resolve_manifest_uri(normalize_manifest_ref(ref), project_root)
    except ValueError as exc:
        click.secho(f"  ✗ {exc}", fg='red')
        return False

    target_id = f'{SHA256_PREFIX}{candidate.stem}'

    try:
        index = load_index(project_root)
    except ValueError as exc:
        click.secho(f"  ✗ {exc}", fg='red')
        return False

    by_id: dict[str, dict[str, Any]] = {
        m['id']: m for m in index['manifests']
        if is_mapping(m) and isinstance(m.get('id'), str)
    }

    chain: list[tuple[str, dict[str, Any] | None]] = []
    seen: set[str] = set()
    current: str | None = target_id
    while current is not None and current not in seen:
        seen.add(current)
        entry = by_id.get(current)
        chain.append((current, entry))
        current = entry.get('parent_id') if isinstance(entry, dict) else None

    chain.reverse()

    target_name = _name_of(by_id.get(target_id), fallback=candidate.stem)
    click.echo(f"Lineage for {SHA256_PREFIX}{short_id(target_id)} ({target_name}):\n")

    for depth, (manifest_id, entry) in enumerate(chain):
        name = _name_of(entry, fallback='(not in store)')
        committed_at = entry.get('committed_at', '') if isinstance(entry, dict) else ''
        connector = '' if depth == 0 else '└─ '
        indent = '  ' + '   ' * depth
        marker = '  ← target' if manifest_id == target_id else ''
        click.echo(f"{indent}{connector}{SHA256_PREFIX}{short_id(manifest_id)}  {name:<28}  {committed_at}{marker}")

    return True


def _name_of(entry: dict[str, Any] | None, fallback: str) -> str:

    if isinstance(entry, dict) and isinstance(entry.get('name'), str):
        return entry['name']
    return fallback
