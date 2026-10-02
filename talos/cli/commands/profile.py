"""Report experiment search-space and sampled runtime estimates."""

from pathlib import Path

import click

from talos.cli.commands._load_yaml import load_and_validate
from talos.yaml.compiler import CompiledSFD
from talos.yaml.profiler import ProfileResult, profile

_COMPLEXITY_COLOURS = {
    'low': 'green',
    'medium': 'yellow',
    'high': 'red',
    'extreme': 'red',
}


def run_profile(yaml_path: Path) -> bool:

    '''
    Validate, compile, and profile a YAML experiment file.

    Args:
        yaml_path (Path): Path to the YAML experiment file

    Returns:
        bool: True on success, False on validation failure or profile error

    '''

    click.echo(f"Profiling {yaml_path} ...")

    yaml_dict, valid = load_and_validate(yaml_path)
    if not valid:
        return False

    try:
        sfd = CompiledSFD(yaml_dict, source_path=yaml_path)
        prof = profile(sfd)
    except Exception as exc:  # noqa: BLE001
        click.secho(f'  ✗ Profile failed: {exc}', fg='red')
        return False

    _print_profile(prof)
    return True


def _print_profile(prof: ProfileResult) -> None:

    click.echo('')
    _print_permutation_space(prof)
    _print_runtime_sampling(prof)
    _print_data_quality(prof)

    for w in prof.warnings:
        click.secho(f'  WARN: {w}', fg='yellow')
    for e in prof.errors:
        click.secho(f'  ERROR: {e}', fg='red')


def _print_permutation_space(prof: ProfileResult) -> None:

    colour = _COMPLEXITY_COLOURS.get(prof.complexity_rating, 'white')
    rating_label = click.style(f'[{prof.complexity_rating}]', fg=colour)
    n_params = len(prof.param_cardinalities)
    click.echo("  Permutation space")
    click.echo(f"    Total:       {format_space(prof.total_permutations)}  {rating_label}")
    click.echo(f"    Parameters:  {n_params}")

    sorted_params = sorted(
        prof.param_cardinalities.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for name, card in sorted_params:
        click.echo(f"      {name:<24} {card} value{'s' if card != 1 else ''}")


def _print_data_quality(prof: ProfileResult) -> None:

    click.echo('')
    click.echo('  Data quality')
    if not prof.data_quality_warnings:
        click.echo('    No NaN or Inf detected in sampled caller-prepared numeric arrays')
        return
    for w in prof.data_quality_warnings:
        click.secho(f'    WARN: {w}', fg='yellow')


def _print_runtime_sampling(prof: ProfileResult) -> None:

    click.echo('')
    attempted = prof.sample_permutations_attempted
    completed = prof.sample_permutations_completed

    if attempted == 0:
        click.echo('  Runtime sampling')
        if prof.errors:
            click.echo('    No completed SFD samples (see errors below)')
        else:
            click.echo('    Skipped — no permutations in covering array')
        return

    click.echo(f"  Runtime sampling  ({completed} of {attempted} completed)")

    if prof.sample_time_seconds_per_permutation is not None:
        t = prof.sample_time_seconds_per_permutation
        click.echo(f"    Per permutation: {t:.3f}s")
        estimated_total = t * prof.total_permutations
        click.echo(f"    Estimated total: {_format_duration(estimated_total)}  ({format_space(prof.total_permutations)} permutations)")
    else:
        click.echo('    Per permutation: —  (no completed runs)')


_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400
_SECONDS_PER_YEAR = 31_536_000

_SPACE_UNITS = [
    (10 ** 12, 'T'),
    (10 ** 9, 'B'),
    (10 ** 6, 'M'),
    (10 ** 3, 'K'),
]


def format_space(n: int) -> str:

    if n >= 10 ** 15:
        return f"{float(n):.2e}"
    for threshold, suffix in _SPACE_UNITS:
        if n >= threshold:
            return f"{n / threshold:.2f}{suffix}"
    return str(n)


def _format_duration(seconds: float) -> str:

    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    if seconds < _SECONDS_PER_HOUR:
        return f"{seconds / _SECONDS_PER_MINUTE:.1f}m"
    if seconds < _SECONDS_PER_DAY:
        return f"{seconds / _SECONDS_PER_HOUR:.1f}h"
    if seconds < _SECONDS_PER_YEAR:
        return f"{seconds / _SECONDS_PER_DAY:.1f}d"
    return f"{seconds / _SECONDS_PER_YEAR:.1f}y"
