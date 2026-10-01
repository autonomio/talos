"""Validate declarative manifest fields before resolving trusted caller code."""
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from talos.yaml._validation import sfd_errors, uel_errors
from talos.yaml.errors import ValidationError, YAMLError
from talos.yaml.schema import VALID_MODES, VERSION


@dataclass
class ValidationResult:
    valid: bool
    errors: list[YAMLError] = field(default_factory=list[YAMLError])
    warnings: list[YAMLError] = field(default_factory=list[YAMLError])
    mode: str = 'development'


def validate(document: object) -> ValidationResult:
    """Return field-specific errors without importing or invoking the SFD."""
    if not isinstance(document, dict):
        return ValidationResult(False, [YAMLError('Manifest must be a mapping')])
    manifest = cast(Mapping[str, object], document)
    errors = [YAMLError('Unknown manifest field', path=key)
              for key in set(manifest) - {'schema_version', 'metadata', 'sfd', 'uel', 'lineage'}]
    if manifest.get('schema_version') != VERSION:
        errors.append(YAMLError(f'Expected schema_version: "{VERSION}"', path='schema_version'))
    metadata = manifest.get('metadata', {})
    if not isinstance(metadata, dict):
        errors.append(YAMLError('Must be a mapping', path='metadata'))
        metadata = {}
    fields = cast(Mapping[str, object], metadata)
    name = fields.get('name')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        errors.append(YAMLError('Use a nonempty name containing letters, digits, underscores or hyphens', path='metadata.name'))
    mode = fields.get('mode', 'development')
    if not isinstance(mode, str) or mode not in VALID_MODES:
        errors.append(YAMLError('Expected development or production', path='metadata.mode'))
    errors.extend(sfd_errors(manifest.get('sfd', {})))
    errors.extend(uel_errors(manifest.get('uel', {})))
    return ValidationResult(not errors, errors, mode=mode if isinstance(mode, str) else 'development')


def validate_or_raise(document: object) -> None:
    """Stop compilation before caller imports when manifest controls are invalid."""
    outcome = validate(document)
    if not outcome.valid:
        raise ValidationError(outcome.errors)
