from datetime import datetime
from datetime import timezone
from pathlib import Path

import click

from talos.cli.git_utils import git_push
from talos.cli.git_utils import git_snapshot
from talos.yaml.config import find_project_root
from talos.yaml.config import read_talos_toml


def run_backup(start: Path) -> bool:

    '''
    Snapshot the project and push it to the configured backup remote.

    Stages and commits the current state (honoring .gitignore, so development
    runs under results/dev/ are never backed up), then pushes to backup_remote.

    Args:
        start (Path): Directory to start searching for the project root

    Returns:
        bool: True on success, False on failure

    '''

    project_root = find_project_root(start)
    if project_root is None:
        click.secho('  ✗ No talos project found. Run this command from inside a Talos project.', fg='red')
        return False

    backup_remote = _resolve_backup_remote(project_root)
    if backup_remote is None:
        return False

    try:
        snapshot_message = f"backup: snapshot {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
        snapshotted, snapshot_error = git_snapshot(project_root, snapshot_message)
        if not snapshotted:
            click.secho(f"  ✗ Could not snapshot project state: {snapshot_error}", fg='red')
            return False

        click.echo(f"Backing up to {backup_remote} ...")
        ok, error = git_push(project_root, backup_remote)
    except FileNotFoundError:
        click.secho('  ✗ git not found on PATH — install git and try again.', fg='red')
        return False

    if not ok:
        click.secho(f"  ✗ Backup failed: {error}", fg='red')
        _print_failure_hint(error, backup_remote)
        return False

    click.secho('  ✓ Backed up', fg='green')
    return True


def _resolve_backup_remote(project_root: Path) -> str | None:

    try:
        config = read_talos_toml(project_root)
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"  ✗ Cannot read talos.toml: {exc}", fg='red')
        return None

    backup_remote = config.get('store', {}).get('backup_remote', '')
    if not isinstance(backup_remote, str) or not backup_remote:
        click.secho(
            '  ✗ No backup remote configured.\n    Set backup_remote in talos.toml, e.g. backup_remote = "git@github.com:user/my-project.git"',
            fg='red',
        )
        return None
    return backup_remote


def _print_failure_hint(error: str, backup_remote: str) -> None:

    if _is_https_credential_error(error):
        click.secho(
            '    HTTPS remote needs credentials — run `gh auth setup-git`, configure a git credential helper, or use an SSH URL (git@github.com:user/repo.git).',
            fg='yellow',
        )
    elif _is_diverged_error(error):
        click.secho(
            f"    The backup remote has history that is not in your project, and talos will not force-push.\n    Back up to an empty repository, or take manual control:\n      git push {backup_remote} HEAD          # add --force only to overwrite the remote",
            fg='yellow',
        )


def _is_https_credential_error(error: str) -> bool:

    lowered = error.lower()
    return 'could not read username' in lowered or 'authentication failed' in lowered


def _is_diverged_error(error: str) -> bool:

    lowered = error.lower()
    return (
        'non-fast-forward' in lowered
        or 'fetch first' in lowered
        or 'failed to push some refs' in lowered
        or 'updates were rejected' in lowered
    )
