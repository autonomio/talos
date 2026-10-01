from dataclasses import dataclass, field
import re
from talos.yaml.errors import YAMLError
from talos.yaml.schema import VERSION, VALID_MODES


@dataclass
class ValidationResult:
    valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    mode: str = 'development'


def validate(document):
    errors = []
    def issue(path, message):
        errors.append(YAMLError(message, path=path))
    if not isinstance(document, dict):
        return ValidationResult(False, [YAMLError('Manifest must be a mapping')])
    for key in set(document) - {'schema_version', 'metadata', 'sfd', 'uel', 'lineage'}:
        issue(key, 'Unknown manifest field')
    if document.get('schema_version') != VERSION:
        issue('schema_version', f'Expected schema_version: "{VERSION}"')
    metadata = document.get('metadata', {})
    if not isinstance(metadata, dict):
        issue('metadata', 'Must be a mapping'); metadata = {}
    name = metadata.get('name')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        issue('metadata.name', 'Use a nonempty name containing letters, digits, underscores or hyphens')
    mode = metadata.get('mode', 'development')
    if mode not in VALID_MODES:
        issue('metadata.mode', 'Expected development or production')
    sfd = document.get('sfd', {})
    if not isinstance(sfd, dict):
        issue('sfd', 'Must be a mapping'); sfd = {}
    if not isinstance(sfd.get('module'), str) or not sfd.get('module'):
        issue('sfd.module', 'Required caller SFD module name or project .py path')
    for key in set(sfd) - {'module', 'params', 'task', 'objective', 'backend', 'context'}:
        issue(f'sfd.{key}', 'Unknown SFD field; data acquisition belongs in caller prep/entrypoint')
    params = sfd.get('params', {})
    if not isinstance(params, dict):
        issue('sfd.params', 'Must be a parameter mapping')
    else:
        for key, values in params.items():
            if not isinstance(values, (list, tuple, range)) or not len(values):
                issue(f'sfd.params.{key}', 'Each parameter requires a nonempty sequence')
    uel = document.get('uel', {})
    if not isinstance(uel, dict):
        issue('uel', 'Must be a mapping'); uel = {}
    allowed = {'n_permutations','round_limit','seed','search_strategy','pruning_strategies','feedback_interval','checkpoint_interval','output_format','output_path','prep_each_round','intra_callback','objective','backend','progress_bar','time_limit','performance_target','save_models','context'}
    for key in set(uel) - allowed:
        issue(f'uel.{key}', 'Unknown experiment setting')
    for key in ('n_permutations','round_limit','feedback_interval','checkpoint_interval'):
        if key in uel and (isinstance(uel[key], bool) or not isinstance(uel[key], int) or uel[key] < 1):
            issue(f'uel.{key}', 'Must be a positive integer')
    strategy = uel.get('search_strategy', {})
    if not isinstance(strategy, dict) or strategy.get('type', 'random') not in {'random', 'grid'}:
        issue('uel.search_strategy', 'Expected random or grid strategy mapping')
    if uel.get('output_format', 'csv') not in {'csv', 'parquet'}:
        issue('uel.output_format', 'Expected csv or parquet')
    if not isinstance(uel.get('pruning_strategies', []), list):
        issue('uel.pruning_strategies', 'Must be a list')
    return ValidationResult(not errors, errors, mode=mode)
