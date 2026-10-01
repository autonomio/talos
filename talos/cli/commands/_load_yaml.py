from pathlib import Path
from typing import Any

import click

from talos.yaml.parser import parse
from talos.yaml.validator import validate as _validate


def load_and_validate(yaml_path: Path) -> tuple[dict[str, Any], bool]:

    '''
    Parse and validate a YAML experiment file, printing all errors and warnings.

    Args:
        yaml_path (Path): Path to the YAML experiment file

    Returns:
        tuple[dict, bool]: (yaml_dict, valid). When valid is False, errors have
            already been printed and the caller should abort.

    '''

    yaml_dict, parse_errors = parse(yaml_path)
    if parse_errors:
        for e in parse_errors:
            location = f' (line {e.line})' if e.line else ''
            click.secho(f'  PARSE ERROR{location}: {e.message}', fg='red')
        return yaml_dict, False

    result = _validate(yaml_dict)
    for e in result.errors:
        location = f' (line {e.line})' if e.line else ''
        path = f'  [{e.path}]' if e.path else ''
        suggestion = f'\n    → {e.suggestion}' if e.suggestion else ''
        click.secho(f'  ERROR{path}{location}: {e.message}{suggestion}', fg='red')
    for w in result.warnings:
        location = f' (line {w.line})' if w.line else ''
        path = f'  [{w.path}]' if w.path else ''
        click.secho(f'  WARN{path}{location}: {w.message}', fg='yellow')

    if not result.valid:
        click.secho(f'  ✗ {len(result.errors)} error(s) found', fg='red')
        return yaml_dict, False

    click.secho('  ✓ Valid', fg='green')
    return yaml_dict, True
