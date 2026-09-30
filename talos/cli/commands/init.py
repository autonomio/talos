import re
import shutil
from pathlib import Path

import click

from talos.cli.commands._constants import TEMPLATES_DIR
from talos.yaml.config import find_project_root

_NAME_SLUG_RE = re.compile(r'^[A-Za-z0-9_-]+$')


def _resolve_output(output: Path) -> Path:
    if output.suffix != '.yaml':
        output = output.with_suffix('.yaml')
    if output.parent == Path('.'):
        project_root = find_project_root(Path.cwd())
        if project_root is not None:
            return project_root / 'manifests' / output.name
        click.secho(
            'Warning: no talos.toml found — creating in current directory. Run from inside a Talos project to place files in manifests/ automatically.',
            fg='yellow',
        )
    return output


def run_init(output: Path, template_name: str | None) -> bool:

    '''
    Scaffold a new YAML experiment file from a template.

    Args:
        output (Path): Destination path for the new experiment file
        template_name (str | None): Template stem name (e.g. 'tf_keras'); if None,
            lists available templates and returns False

    Returns:
        bool: True on success, False on failure

    '''

    output = _resolve_output(output)
    available = {p.stem: p for p in sorted(TEMPLATES_DIR.glob('*.yaml'))}

    if template_name is None:
        click.echo('Available templates:')
        for name in available:
            marker = '  (default)' if name == 'tf_keras' else ''
            click.echo(f'  {name}{marker}')
        click.echo()
        template_name = click.prompt('Template', default='tf_keras')
        if template_name not in available:
            click.secho(
                f"Template '{template_name}' not found. Available: {', '.join(available)}",
                fg='red',
            )
            return False

    if template_name not in available:
        click.secho(
            f"Template '{template_name}' not found. Available: {', '.join(available)}",
            fg='red',
        )
        return False

    if output.exists():
        click.secho(f"'{output}' already exists — will not overwrite.", fg='red')
        return False

    experiment_name = output.stem
    if not _NAME_SLUG_RE.match(experiment_name):
        click.secho(
            f"  ✗ '{experiment_name}' is not a valid experiment name. Use only letters, digits, underscores, or hyphens.",
            fg='red',
        )
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    _ = shutil.copy2(available[template_name], output)

    module_path = output.parent / f'{template_name}_sfd.py'
    source = Path(__file__).resolve().parents[2] / 'sfd' / 'templates' / f'{template_name}.py'
    if not module_path.exists():
        shutil.copy2(source, module_path)
    text = output.read_text().replace(f'talos.sfd.templates.{template_name}', f'{template_name}_sfd.py')
    output.write_text(text)

    if not _update_experiment_name(output, experiment_name):
        click.secho(
            f"  ✗ Template '{template_name}' has no metadata.name field to update.",
            fg='red',
        )
        output.unlink()
        return False

    click.secho(f"  ✓ Created '{output}' from template '{template_name}'", fg='green')
    click.echo(f"  Edit the file, then run: talos validate '{output}'")
    return True


def _update_experiment_name(path: Path, name: str) -> bool:

    text = path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    in_metadata = False
    replaced = False

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        if not indent and stripped.rstrip('\n') == 'metadata:':
            in_metadata = True
            continue

        if not indent and stripped and not stripped.startswith('#'):
            in_metadata = False

        if in_metadata and stripped.startswith('name:'):
            lines[i] = f'{indent}name: "{name}"\n'
            replaced = True
            break

    if replaced:
        _ = path.write_text(''.join(lines), encoding='utf-8')
    return replaced
