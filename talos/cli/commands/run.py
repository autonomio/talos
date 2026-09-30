import math
import shutil
from datetime import datetime
from pathlib import Path
import click
from talos.cli.commands._load_yaml import load_and_validate
from talos.cli.commands.profile import format_space
from talos.yaml.compiler import CompiledSFD, build_search_strategy

DEFAULT_RESULTS_BASE = Path('.')


def run_experiment(yaml_path, dry_run=False, manifest_id=None,
                   results_base=DEFAULT_RESULTS_BASE, progress_bar=True):
    """Execute caller-owned params/prep/model code from a validated manifest."""
    click.echo(f"Loading {yaml_path} ...")
    document, valid = load_and_validate(yaml_path)
    if not valid:
        return False
    try:
        compiled = CompiledSFD(document, source_path=yaml_path)
        parameters = compiled.params()
        build_search_strategy(document, parameters)
        compiled.pruning_strategies()
    except Exception as exc:
        click.secho(f'  Compilation failed: {exc}', fg='red')
        return False
    if dry_run:
        click.echo('  Dry run complete: SFD and parameter references resolved')
        return True
    settings = document.get('uel', {})
    name = document['metadata']['name']
    development = document['metadata'].get('mode', 'development') == 'development'
    output = _build_results_dir(settings, name, development, manifest_id, results_base)
    output.mkdir(parents=True, exist_ok=True)
    click.echo(f"Running '{name}' — {format_space(math.prod(len(v) for v in parameters.values()))} combinations")
    click.echo(f'  Results → {output}')
    try:
        result = compiled.execute(experiment_dir=output, progress_bar=progress_bar)
        shutil.copy2(yaml_path, output / ('manifest.yaml' if manifest_id else Path(yaml_path).name))
    except Exception as exc:
        click.secho(f'  Experiment failed: {exc}', fg='red')
        return False
    click.secho('  Experiment complete', fg='green')
    return True


def _build_results_dir(uel_cfg, experiment_name, test_mode, manifest_id=None,
                       results_base=DEFAULT_RESULTS_BASE):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    prefix = Path(results_base) / 'results'
    if test_mode:
        prefix /= 'dev'
    if manifest_id:
        return prefix / manifest_id[:8] / timestamp
    template = uel_cfg.get('output_path', '{name}_{datetime}')
    return prefix / template.replace('{name}', experiment_name).replace('{datetime}', timestamp)
