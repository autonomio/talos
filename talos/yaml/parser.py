from pathlib import Path
from typing import Any

from ruamel.yaml.constructor import DuplicateKeyError

from talos.yaml.config import is_mapping
from talos.yaml.config import round_trip_yaml
from talos.yaml.errors import YAMLError


def parse(source: str | Path) -> tuple[dict[str, Any], list[YAMLError]]:

    '''
    Parse a YAML experiment file or string into a dict.

    Args:
        source (str | Path): Path instance to read from disk, or raw YAML string

    Returns:
        tuple[dict, list[YAMLError]]: Parsed dict and list of parse errors (empty on success)

    '''

    errors: list[YAMLError] = []
    yaml = round_trip_yaml(preserve_quotes=True)

    try:
        if isinstance(source, Path):
            content = source.read_text(encoding='utf-8')
        else:
            content = source

        result = yaml.load(content)

        if result is None:
            errors.append(YAMLError(
                message='YAML file is empty',
                path='',
                suggestion='Add required fields: schema_version, metadata, sfd, uel',
            ))
            return {}, errors

        if not is_mapping(result):
            errors.append(YAMLError(
                message=f'Expected a YAML mapping at root, got {type(result).__name__}',
                path='',
            ))
            return {}, errors

        return dict(result), errors

    except DuplicateKeyError as exc:
        errors.append(YAMLError(
            message=f'Duplicate key: {exc}',
            path='',
        ))
        return {}, errors

    except Exception as exc:  # noqa: BLE001
        mark = getattr(exc, 'problem_mark', None)
        line = (mark.line + 1) if mark else None
        col = (mark.column + 1) if mark else None
        errors.append(YAMLError(
            message=str(exc),
            path='',
            line=line,
            column=col,
        ))
        return {}, errors
