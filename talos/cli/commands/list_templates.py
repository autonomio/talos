import click

from talos.cli.commands._constants import TEMPLATES_DIR
from talos.yaml.parser import parse


def run_list_templates() -> None:

    '''Print all available YAML experiment templates with their descriptions.'''

    templates = sorted(TEMPLATES_DIR.glob('*.yaml'))

    if not templates:
        click.echo('No templates found.')
        return

    click.echo(f'Available templates ({TEMPLATES_DIR}):')
    click.echo()

    name_width = max(len(t.stem) for t in templates)

    for path in templates:
        yaml_dict, errors = parse(path)
        description = ''
        if not errors:
            description = yaml_dict.get('metadata', {}).get('description', '')
        click.echo(f'  {path.stem:<{name_width}}    {description}')
