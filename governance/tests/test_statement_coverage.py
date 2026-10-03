"""Coverage evidence must measure the complete package without rounding up."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools.check_statement_coverage import check_statement_coverage


def _report(tmp_path: Path, covered: int, total: int) -> tuple[Path, Path]:
    package = tmp_path / 'talos'
    package.mkdir()
    (package / 'model.py').write_text('value = 1\n', encoding='utf-8')
    counts = {'covered_lines': covered, 'num_statements': total}
    payload = {'files': {'talos/model.py': {'summary': counts}}, 'totals': counts}
    report = tmp_path / 'coverage.json'
    report.write_text(json.dumps(payload), encoding='utf-8')
    return report, package


def test_statement_threshold_is_independent_of_branch_percentage(tmp_path: Path) -> None:
    report, package = _report(tmp_path, 4, 5)
    payload = json.loads(report.read_text(encoding='utf-8'))
    payload['totals']['percent_covered'] = 40.0
    report.write_text(json.dumps(payload), encoding='utf-8')
    assert check_statement_coverage(report, package) == (4, 5)


def test_near_threshold_cannot_round_up(tmp_path: Path) -> None:
    report, package = _report(tmp_path, 7999, 10000)
    with pytest.raises(ValueError, match='below 80%'):
        check_statement_coverage(report, package)


def test_missing_package_module_is_rejected(tmp_path: Path) -> None:
    report, package = _report(tmp_path, 4, 5)
    (package / 'unmeasured.py').write_text('value = 2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='every package module'):
        check_statement_coverage(report, package)


def test_foreign_copy_cannot_replace_the_original_module(tmp_path: Path) -> None:
    report, package = _report(tmp_path, 4, 5)
    payload = json.loads(report.read_text(encoding='utf-8'))
    payload['files']['copy/model.py'] = payload['files'].pop('talos/model.py')
    report.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='every package module'):
        check_statement_coverage(report, package)


def test_inflated_totals_are_rejected(tmp_path: Path) -> None:
    report, package = _report(tmp_path, 3, 5)
    payload = json.loads(report.read_text(encoding='utf-8'))
    payload['totals']['covered_lines'] = 5
    report.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='totals must match'):
        check_statement_coverage(report, package)


def test_ci_combine_preserves_data_after_nested_report(tmp_path: Path) -> None:
    """Execute both phases with the real config and the final CI combine command."""
    root = Path(__file__).resolve().parents[2]
    package = tmp_path / 'talos'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'model.py').write_text(
        "def first():\n    return 'first'\n\ndef later():\n    return 'later'\n", encoding='utf-8')
    for name in ('first', 'later'):
        (tmp_path / f'{name}.py').write_text(
            f"from talos.model import {name}\nassert {name}() == {name!r}\n", encoding='utf-8')
    env = dict(os.environ)
    env.update(TALOS_COVERAGE_ROOT=str(tmp_path), COVERAGE_FILE=str(tmp_path / '.coverage'),
               COVERAGE_RCFILE=str(root / '.github/coverage-full.toml'), PYTHONPATH=str(tmp_path))

    def coverage(*arguments: str) -> None:
        subprocess.run([sys.executable, '-m', 'coverage', *arguments], cwd=tmp_path,
                       env=env, check=True, capture_output=True, text=True)

    coverage('run', str(tmp_path / 'first.py'))
    coverage('report')
    assert (tmp_path / '.coverage').is_file()
    assert not list(tmp_path.glob('.coverage.*'))
    coverage('run', str(tmp_path / 'later.py'))
    workflow = yaml.safe_load((root / '.github/workflows/ci.yml').read_text(encoding='utf-8'))
    commands = [line.strip() for step in workflow['jobs']['documentation']['steps']
                for line in step.get('run', '').splitlines()
                if line.strip().startswith('python -m coverage combine ')]
    assert len(commands) == 2
    assert all(command == 'python -m coverage combine --append --keep' for command in commands)
    coverage(*shlex.split(commands[0])[3:])
    report = tmp_path / 'coverage.json'
    coverage('json', '-o', str(report))
    assert check_statement_coverage(report, package) == (4, 4)
    measured = json.loads(report.read_text(encoding='utf-8'))['files']['talos/model.py']
    assert {2, 5} <= set(measured['executed_lines'])
