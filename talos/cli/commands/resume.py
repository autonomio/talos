import json
from pathlib import Path
import click
from talos.yaml.compiler import CompiledSFD
from talos.yaml.store import canonical_manifest_id


def run_resume(results_dir, progress_bar=True):
    """Resume the recorded manifest and checkpoint without reconstructing data loaders."""
    results_dir = Path(results_dir)
    click.echo(f'Resuming from {results_dir} ...')
    reference = _load_yaml_reference(results_dir)
    if reference is None:
        return False
    try:
        compiled = CompiledSFD.from_run(results_dir)
        compiled.execute(resume=True, experiment_dir=results_dir, progress_bar=progress_bar)
    except Exception as exc:
        click.secho(f'  Resume failed: {exc}', fg='red')
        return False
    click.secho('  Experiment complete', fg='green')
    return True


def _load_yaml_reference(results_dir):
    try:
        metadata = json.loads((Path(results_dir) / 'metadata.json').read_text())
        reference = metadata['yaml_reference']
        content = reference['content']
        if not isinstance(content, dict):
            raise ValueError('Recorded manifest must be a mapping')
        if reference.get('manifest_id') != canonical_manifest_id(content):
            raise ValueError('Recorded manifest content does not match its hash')
        source = reference.get('source_path')
        if source is not None and not isinstance(source, str):
            raise ValueError('Recorded manifest source path must be a string')
        return reference
    except (OSError, ValueError, KeyError, TypeError) as exc:
        click.secho(f'  Cannot load recorded manifest: {exc}', fg='red')
        return None
