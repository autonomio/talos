"""Mutation tests for every bloat gate: prove each gate fires on its violation."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
GOVERNANCE_DIR: Final[Path] = REPO_ROOT / 'governance'


def _clone_script_into(root: Path, name: str) -> Path:
    dest_dir = root / 'governance'
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / '__init__.py').write_text('', encoding='utf-8')
    # Gates import the shared helper by bare name, so a cloned gate needs
    # _common.py beside it to run in isolation.
    shutil.copy2(GOVERNANCE_DIR / '_common.py', dest_dir / '_common.py')
    shutil.copy2(GOVERNANCE_DIR / '_quality.py', dest_dir / '_quality.py')
    dest = dest_dir / name
    shutil.copy2(GOVERNANCE_DIR / name, dest)
    return dest


def _run(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def _write_budget(root: Path, section: str, value: object) -> None:
    # The budgets share one file now, so each helper merges its own section
    # instead of writing the file whole and clobbering the others.
    github = root / '.github'
    github.mkdir(parents=True, exist_ok=True)
    path = github / 'budgets.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    data[section] = value
    path.write_text(json.dumps(data), encoding='utf-8')


def _write_typing_budget(root: Path, package: str = 'talos') -> None:
    # The bloat gates resolve their scan target from this single source
    # and fail closed without it. A synthetic repo must declare it.
    (root / 'governance.yml').write_text(
        yaml.safe_dump({'layout': {'package_root': package}}), encoding='utf-8',
    )


def test_module_budget_mutation_fires(tmp_path: Path) -> None:
    (tmp_path / '.github').mkdir()
    (tmp_path / 'talos').mkdir()
    _write_typing_budget(tmp_path)
    budget = {'talos/oversized.py': 5}
    _write_budget(tmp_path, 'modules', budget)
    (tmp_path / 'talos' / 'oversized.py').write_text(
        'a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n',  # 6 SLOC > 5 budget
        encoding='utf-8',
    )
    script = _clone_script_into(tmp_path, 'check_module_budgets.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'MODULE BUDGET GATE -- FAIL' in result.stderr
    assert 'oversized.py' in result.stderr


def test_module_docstring_mutation_fires(tmp_path: Path) -> None:
    (tmp_path / 'talos').mkdir()
    _write_typing_budget(tmp_path)
    (tmp_path / 'talos' / 'no_docstring.py').write_text(
        'x = 1\n',
        encoding='utf-8',
    )
    script = _clone_script_into(tmp_path, 'check_module_docstrings.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'MODULE DOCSTRING GATE -- FAIL' in result.stderr
    assert 'no_docstring.py' in result.stderr


def test_module_docstring_multiline_mutation_fires(tmp_path: Path) -> None:
    (tmp_path / 'talos').mkdir()
    _write_typing_budget(tmp_path)
    (tmp_path / 'talos' / 'multiline.py').write_text(
        '"""Summary line.\n\nDetailed paragraph."""\nx = 1\n',
        encoding='utf-8',
    )
    script = _clone_script_into(tmp_path, 'check_module_docstrings.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'MODULE DOCSTRING GATE -- FAIL' in result.stderr
    assert 'multiline.py' in result.stderr


def test_file_size_balance_mutation_fires(tmp_path: Path) -> None:
    (tmp_path / 'talos').mkdir()
    _write_typing_budget(tmp_path)
    (tmp_path / 'talos' / 'small_a.py').write_text('a = 1\n', encoding='utf-8')
    (tmp_path / 'talos' / 'small_b.py').write_text('b = 2\n', encoding='utf-8')
    # Median will be ~1 line; a 10-line file gives ratio > 2.5.
    (tmp_path / 'talos' / 'giant.py').write_text('\n' * 20, encoding='utf-8')
    script = _clone_script_into(tmp_path, 'check_file_size_balance.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'FILE SIZE BALANCE GATE -- FAIL' in result.stderr
    assert 'giant.py' in result.stderr


def test_test_code_ratio_mutation_fires(tmp_path: Path) -> None:
    # source with enough SLOC to bypass the vacuous-pass boundary (50),
    # but tests too small -> ratio < 0.60.
    source_dir = tmp_path / 'talos'
    source_dir.mkdir()
    _write_typing_budget(tmp_path)
    (source_dir / 'big.py').write_text('\n'.join(f'x_{i} = {i}' for i in range(60)) + '\n', encoding='utf-8')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / 'tiny.py').write_text('y = 1\n', encoding='utf-8')
    script = _clone_script_into(tmp_path, 'check_test_code_ratio.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'TEST/CODE RATIO GATE -- FAIL' in result.stderr


def _write_coverage_budget(root: Path, line: int = 50, branch: int = 45) -> None:
    _write_budget(root, 'coverage', {'line': line, 'branch': branch})


def test_coverage_floor_mutation_fires(tmp_path: Path) -> None:
    # Actual coverage below both the line floor (50%) and branch floor (45%).
    _write_coverage_budget(tmp_path)
    coverage = {
        'totals': {
            'num_statements': 100,
            'num_branches': 20,
            'percent_statements_covered': 40.0,
            'percent_branches_covered': 35.0,
        },
        'files': {'talos/foo.py': {'missing_lines': [10, 11, 12]}},
    }
    (tmp_path / 'coverage.json').write_text(json.dumps(coverage), encoding='utf-8')
    script = _clone_script_into(tmp_path, 'check_coverage_floor.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'COVERAGE FLOOR GATE -- FAIL' in result.stderr
    assert '40.0%' in result.stderr


def test_coverage_floor_track_mutation_fires(tmp_path: Path) -> None:
    # Actual coverage clears the floor but has run far ahead of it: the
    # TRACK rule demands the gain be banked, so the gate fires.
    _write_coverage_budget(tmp_path, line=50, branch=45)
    coverage = {
        'totals': {
            'num_statements': 100,
            'num_branches': 30,
            'percent_statements_covered': 92.0,
            'percent_branches_covered': 90.0,
        },
        'files': {},
    }
    (tmp_path / 'coverage.json').write_text(json.dumps(coverage), encoding='utf-8')
    script = _clone_script_into(tmp_path, 'check_coverage_floor.py')
    result = _run(script, cwd=tmp_path)
    assert result.returncode == 1
    assert 'COVERAGE FLOOR GATE -- FAIL' in result.stderr
    assert 'Bank the gain' in result.stderr


def test_coverage_ratchet_mutation_fires(tmp_path: Path) -> None:
    # Head floor is lowered below the base floor with no marker -> fires.
    _write_coverage_budget(tmp_path, line=50, branch=45)
    base_file = tmp_path / 'base.json'
    base_file.write_text(
        json.dumps({'coverage': {'line': 80, 'branch': 70}}), encoding='utf-8')
    body_file = tmp_path / 'body.txt'
    body_file.write_text('', encoding='utf-8')
    script = _clone_script_into(tmp_path, 'check_coverage_ratchet.py')
    result = _run(script, '--base-file', str(base_file), '--pr-body-file', str(body_file), cwd=tmp_path)
    assert result.returncode == 1
    assert 'COVERAGE RATCHET GATE -- FAIL' in result.stderr
    assert 'lowered without marker' in result.stderr


def test_budget_ratchet_mutation_fires(tmp_path: Path) -> None:
    (tmp_path / '.github').mkdir()
    head = {'talos/foo.py': 200}
    base = {'talos/foo.py': 100}
    _write_budget(tmp_path, 'modules', head)
    base_file = tmp_path / 'base.json'
    base_file.write_text(json.dumps({'modules': base}), encoding='utf-8')
    body_file = tmp_path / 'body.txt'
    body_file.write_text('No marker present.', encoding='utf-8')
    script = _clone_script_into(tmp_path, 'check_budget_ratchet.py')
    result = _run(script, '--base-file', str(base_file), '--pr-body-file', str(body_file), cwd=tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'BUDGET RATCHET GATE -- FAIL' in result.stderr
    assert 'talos/foo.py' in result.stderr


def test_ruff_complexity_mutation_fires(tmp_path: Path) -> None:
    # A function with cyclomatic complexity > 10 (enough nested branches).
    body = 'def too_complex(x: int) -> int:\n    y: int = 0\n'
    for i in range(15):
        body += f'    if x == {i}:\n        y += {i}\n'
    body += '    return y\n'
    target = tmp_path / 'mutation.py'
    target.write_text(body, encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--isolated',
         '--select', 'C901', str(target)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'C901' in result.stdout + result.stderr


def test_ruff_print_mutation_fires(tmp_path: Path) -> None:
    target = tmp_path / 'mutation.py'
    target.write_text('def f() -> None:\n    print("hello")\n', encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--isolated', '--select', 'T201', str(target)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'T201' in result.stdout + result.stderr


def test_ruff_todo_mutation_fires(tmp_path: Path) -> None:
    target = tmp_path / 'mutation.py'
    target.write_text('# TODO: implement me\nx = 1\n', encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--isolated', '--select', 'FIX002', str(target)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'FIX002' in result.stdout + result.stderr


def test_ruff_commented_code_mutation_fires(tmp_path: Path) -> None:
    target = tmp_path / 'mutation.py'
    # ERA001 catches commented-out code. A pattern like `# x = 1` is flagged.
    target.write_text(textwrap.dedent('''
        def f() -> None:
            # y = compute_something(x, y, z)
            return None
    ''').strip() + '\n', encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--isolated', '--select', 'ERA001', str(target)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'ERA001' in result.stdout + result.stderr


def test_vulture_mutation_fires(tmp_path: Path) -> None:
    source = tmp_path / 'talos'
    source.mkdir()
    (source / '__init__.py').write_text('', encoding='utf-8')
    # Vulture assigns ≥ 90% confidence to unused imports; the ≥ 80%
    # threshold must catch that class. Unused variables and functions
    # sit at 60% and are intentionally below our CI threshold — they
    # require higher confidence to avoid false positives on the
    # production codebase.
    (source / 'bait.py').write_text(
        '"""Bait module."""\n\nimport json  # deliberately unused\n',
        encoding='utf-8',
    )
    result = subprocess.run(
        [sys.executable, '-m', 'vulture', 'talos/', '--min-confidence', '80'],
        check=False, capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode != 0, f'vulture did not flag: {result.stdout!r}'
    assert 'json' in result.stdout
