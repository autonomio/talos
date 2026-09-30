from pathlib import Path

import click

from talos.cli.commands._load_yaml import load_and_validate


def run_validate(yaml_path: Path) -> bool:

    '''
    Parse and validate a YAML experiment file.

    Args:
        yaml_path (Path): Path to the YAML experiment file

    Returns:
        bool: True if valid, False if errors were found

    '''

    click.echo(f"Validating {yaml_path} ...")
    _, valid = load_and_validate(yaml_path)
    return valid
