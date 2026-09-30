from pathlib import Path

import click

from talos.cli.git_utils import git_clone
from talos.cli.git_utils import run_git





def run_new(project_name: str,
            backup_remote: str | None,
            from_remote: str | None = None) -> bool:

    '''
    Create a new Talos project from the template or restore one from backup.

    Args:
        project_name (str): Name of the new project directory to create
        backup_remote (str | None): Git remote URL for manifest store backup
        from_remote (str | None): Backup remote to restore from; when set, the
            project is cloned from it with history intact instead of scaffolded
            from the template

    Returns:
        bool: True on success, False on failure

    '''

    project_path = Path(project_name)

    if project_path.exists():
        click.secho(f"  ✗ '{project_name}' already exists.", fg='red')
        return False

    if from_remote is not None:
        return _restore_from_backup(project_path, from_remote)

    click.echo(f"Creating project '{project_name}' ...")

    if not _scaffold_project(project_path):
        return False

    if backup_remote:
        _write_backup_remote(project_path, backup_remote)

    click.secho(f"\n  ✓ Project '{project_name}' created.", fg='green')
    click.echo("\n  Next steps:")
    click.echo(f"    cd {project_name}")
    click.echo("    talos list-templates")
    click.echo("    talos init first.yaml --template tf_keras")
    click.echo("    talos validate first.yaml")
    return True


def _restore_from_backup(project_path: Path, from_remote: str) -> bool:

    project_name = project_path.name
    click.echo(f"Restoring project '{project_name}' from {from_remote} ...")
    try:
        ok, error = git_clone(from_remote, project_path)
    except FileNotFoundError:
        click.secho('  ✗ git not found on PATH — install git and try again.', fg='red')
        return False
    if not ok:
        click.secho(f"  ✗ Restore failed: {error}", fg='red')
        return False

    if not (project_path / 'talos.toml').exists():
        click.secho(
            f"  ⚠ Restored '{project_name}', but no talos.toml found — this may not be a Talos project backup.\n    If your backup is on another branch, clone it directly:\n      git clone -b <branch> {from_remote} {project_name}",
            fg='yellow',
        )

    click.secho(f"\n  ✓ Project '{project_name}' restored.", fg='green')
    click.echo("\n  Next steps:")
    click.echo(f"    cd {project_name}")
    click.echo("    talos ls")
    return True


def _scaffold_project(project_path: Path) -> bool:
    """Create a local caller-owned project; no external template repository."""
    project_path.mkdir(parents=True)
    (project_path / 'manifests' / 'committed').mkdir(parents=True)
    (project_path / 'talos.toml').write_text('[store]\nbackup_remote = ""\n')
    (project_path / '.gitignore').write_text('results/dev/\n__pycache__/\n.venv/\n')
    (project_path / 'README.md').write_text(
        '# Talos project\n\nUse `talos init first --template tf_keras` (or keras/pytorch).\n'
        'Edit the generated SFD `prep` to supply your own data and `model` to train.\n'
        'Talos owns experiment search, manifests, checkpoints, artifacts and analytics.\n')
    try:
        initialized = run_git(['init'], cwd=project_path)
        if initialized.returncode:
            click.secho(initialized.stderr.strip(), fg='red')
            return False
        run_git(['symbolic-ref', 'HEAD', 'refs/heads/main'], cwd=project_path)
        run_git(['add', '.'], cwd=project_path)
        committed = run_git(['commit', '-m', 'Create Talos project'], cwd=project_path)
        if committed.returncode:
            click.secho('  Project created; initial Git commit requires configured user identity.', fg='yellow')
    except FileNotFoundError:
        click.secho('  Project created; Git is unavailable.', fg='yellow')
    return True


def _write_backup_remote(project_path: Path, remote_url: str) -> None:

    if '"' in remote_url or '\n' in remote_url:
        click.secho('  ⚠ Backup remote URL contains invalid characters — skipping.', fg='yellow')
        return
    toml_path = project_path / 'talos.toml'
    text = toml_path.read_text(encoding='utf-8')
    updated = text.replace('backup_remote = ""', f'backup_remote = "{remote_url}"')
    if updated == text:
        click.secho('  ⚠ Could not set backup remote — talos.toml format unexpected.', fg='yellow')
        return
    _ = toml_path.write_text(updated, encoding='utf-8')
    click.echo(f"  Backup remote set to: {remote_url}")
