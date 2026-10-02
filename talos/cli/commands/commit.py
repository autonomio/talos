"""Validate and commit experiment manifests with content identities."""

from pathlib import Path

import click

from talos.cli.commands._load_yaml import load_and_validate
from talos.cli.git_utils import git_add_and_commit
from talos.yaml.config import find_project_root
from talos.yaml.store import (
    SHA256_PREFIX,
    commit_manifest,
    is_full_manifest_id,
    lineage_block,
    manifest_name,
    short_id,
)


def run_commit(yaml_path: Path, parent_id: str | None, message: str | None) -> bool:

    '''
    Validate, store, and git-commit a YAML manifest.

    Args:
        yaml_path (Path): Path to the source YAML file
        parent_id (str | None): Parent manifest ID for lineage tracking
        message (str | None): Custom git commit message; auto-generated if None

    Returns:
        bool: True on success, False on failure

    '''

    project_root = find_project_root(yaml_path.parent)
    if project_root is None:
        click.secho(
            '  ✗ No talos project found. The YAML file must be inside a Talos project directory (containing talos.toml).',
            fg='red', )
        return False

    click.echo(f"Validating {yaml_path.name} ...")
    yaml_dict, valid = load_and_validate(yaml_path)
    if not valid:
        return False

    if parent_id is None:
        lineage_parent = lineage_block(yaml_dict).get('parent_id')
        if isinstance(lineage_parent, str):
            parent_id = lineage_parent

    if parent_id is not None and not is_full_manifest_id(parent_id):
        click.secho(
            f"  ✗ Invalid parent ID: '{parent_id}'\n    Expected sha256:<64-hex-chars>.",
            fg='red', )
        return False

    mode = yaml_dict.get('metadata', {}).get('mode', 'development')
    if mode != 'production':
        click.secho(
            '  ✗ Cannot commit a development-mode manifest.\n    Set metadata.mode: production before committing.',
            fg='red', )
        return False

    manifest_id, already_existed = commit_manifest(yaml_path, project_root, parent_id)
    short = f'{SHA256_PREFIX}{short_id(manifest_id)}'

    if already_existed:
        repair_msg = f'repair: restore index for {short}'
        if git_add_and_commit(project_root, Path('manifests') / 'committed', repair_msg):
            click.secho(f"\n  ✓ Repaired and committed {manifest_id}", fg='green')
        else:
            click.secho(f"\n  Already in store: {manifest_id}", fg='yellow')
        click.echo(f"  Run with: talos run {manifest_id}")
        return True

    name = manifest_name(yaml_dict, fallback=yaml_path.stem)
    commit_msg = message or f'commit: {name} ({short})'
    git_ok = git_add_and_commit(project_root, Path('manifests') / 'committed', commit_msg)

    if not git_ok:
        click.secho(
            f"\n  ✓ Stored {manifest_id}\n  ⚠ Git commit failed — manifest is stored but not version-controlled.",
            fg='yellow',
        )
    else:
        click.secho(f"\n  ✓ Committed {manifest_id}", fg='green')
    click.echo(f"  Run with: talos run {manifest_id}")
    return True
