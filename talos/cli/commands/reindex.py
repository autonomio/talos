from pathlib import Path

import click

from talos.yaml.config import find_project_root
from talos.yaml.store import rebuild_index


def run_reindex(start: Path) -> bool:

    '''
    Rebuild the manifest store index from the committed manifest files.

    Args:
        start (Path): Directory to start searching for the project root

    Returns:
        bool: True on success, False if no project found

    '''

    project_root = find_project_root(start)
    if project_root is None:
        click.secho('  ✗ No talos project found. Run this command from inside a Talos project.', fg='red')
        return False

    count, warnings = rebuild_index(project_root)

    for warning in warnings:
        click.secho(f"  ⚠ {warning}", fg='yellow')

    click.secho(f"  ✓ Rebuilt index with {count} manifest(s)", fg='green')
    return True
