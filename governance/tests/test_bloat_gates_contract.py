"""Contract tests for the hard-mechanical bloat gates (slice #11)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

from _common import loads_toml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
BUDGET_JSON: Final[Path] = REPO_ROOT / '.github/budgets.json'
BUDGET_SECTION: Final[str] = 'modules'
LINT_WORKFLOW: Final[Path] = REPO_ROOT / '.github/workflows/pr_checks_lint.yml'
GOVERNANCE_DIR: Final[Path] = REPO_ROOT / 'governance'
PYPROJECT: Final[Path] = REPO_ROOT / 'pyproject.toml'

GATE_SCRIPTS: Final[list[str]] = [
    'check_module_budgets.py',
    'check_test_code_ratio.py',
    'check_module_docstrings.py',
    'check_docstrings.py',
    'check_file_size_balance.py',
    'check_coverage_floor.py',
    'check_coverage_ratchet.py',
    'check_dependency_vulnerabilities.py',
    'check_diff_coverage.py',
    'check_budget_ratchet.py',
    'check_no_swallowed_violations.py',
    'check_test_fallbacks.py',
]

GATE_BANNERS: Final[dict[str, str]] = {
    'check_module_budgets.py': 'MODULE BUDGET GATE',
    'check_test_code_ratio.py': 'TEST/CODE RATIO GATE',
    'check_module_docstrings.py': 'MODULE DOCSTRING GATE',
    'check_docstrings.py': 'DOCSTRING CONVENTIONS GATE',
    'check_file_size_balance.py': 'FILE SIZE BALANCE GATE',
    'check_coverage_floor.py': 'COVERAGE FLOOR GATE',
    'check_coverage_ratchet.py': 'COVERAGE RATCHET GATE',
    'check_dependency_vulnerabilities.py': 'DEPENDENCY VULNERABILITY GATE',
    'check_diff_coverage.py': 'DIFF COVERAGE GATE',
    'check_budget_ratchet.py': 'BUDGET RATCHET GATE',
    'check_no_swallowed_violations.py': 'NO SWALLOWED VIOLATIONS GATE',
    'check_test_fallbacks.py': 'TEST FALLBACK GATE',
}


def _run(script: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(GOVERNANCE_DIR / script), *args]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=cwd or REPO_ROOT)


def test_module_budgets_is_valid_json() -> None:
    data = json.loads(BUDGET_JSON.read_text(encoding='utf-8'))[BUDGET_SECTION]
    assert isinstance(data, dict)
    assert all(isinstance(k, str) for k in data)
    assert all(isinstance(v, int) and v > 0 for v in data.values())


def test_module_budgets_covers_every_package_path() -> None:
    data = json.loads(BUDGET_JSON.read_text(encoding='utf-8'))[BUDGET_SECTION]
    package_paths = {p for p in data if p.startswith('talos/')}
    script_paths = {p for p in data if p.startswith('governance/')}
    # Every .py under the package root is declared in
    # budgets.json "modules". Otherwise a new module could silently escape
    # the line-count budget gate.
    actual_paths = _actual_package_paths()
    assert package_paths == actual_paths, (
        f'budgets.json "modules" paths diverge from actual source tree: '
        f'extra={sorted(package_paths - actual_paths)}, '
        f'missing={sorted(actual_paths - package_paths)}'
    )
    # Two invariants, rather than a count that has to be edited whenever a
    # helper is added: no `check_*` gate may escape the budget, and no budget
    # entry may point at a file that no longer exists. Which helpers besides
    # the gates carry a budget is a judgement already made per module, so it
    # is not re-derived here.
    gates = {
        f'governance/{q.name}'
        for q in (REPO_ROOT / 'governance').glob('*.py')
        if q.name.startswith('check_')
    }
    assert gates <= script_paths, f'ungated: {sorted(gates - script_paths)}'
    stale = {q for q in script_paths if not (REPO_ROOT / q).is_file()}
    assert not stale, f'budget entries with no file: {sorted(stale)}'


def _actual_package_paths() -> set[str]:
    root = REPO_ROOT / 'talos'
    return {
        str(p.relative_to(REPO_ROOT)).replace('\\', '/')
        for p in root.rglob('*.py')
        if '__pycache__' not in p.parts
    }


def test_all_gate_scripts_exist_and_are_executable() -> None:
    for name in GATE_SCRIPTS:
        path = GOVERNANCE_DIR / name
        assert path.is_file(), f'{path} missing'
        assert path.stat().st_mode & 0o111, f'{path} not executable'


def test_all_scripts_pass_on_current_repo() -> None:
    _run('check_module_budgets.py').check_returncode()
    _run('check_test_code_ratio.py').check_returncode()
    _run('check_module_docstrings.py').check_returncode()
    _run('check_file_size_balance.py').check_returncode()
    _run(
        'check_budget_ratchet.py',
        '--base-file', '/dev/null',
        '--pr-body-file', '/dev/null',
    ).check_returncode()


def test_pass_banners_printed_on_success() -> None:
    for name in ('check_module_budgets.py', 'check_test_code_ratio.py',
                 'check_module_docstrings.py', 'check_file_size_balance.py'):
        result = _run(name)
        assert result.returncode == 0, result.stderr
        banner = GATE_BANNERS[name]
        assert f'{banner} -- PASS' in result.stdout, f'{name} stdout: {result.stdout!r}'


def test_fail_banners_are_declared_in_each_script_source() -> None:
    for name in GATE_SCRIPTS:
        source = (GOVERNANCE_DIR / name).read_text(encoding='utf-8')
        banner = GATE_BANNERS[name]
        assert f'{banner} -- FAIL' in source, f'{name} missing FAIL banner literal'
        assert f'{banner} -- PASS' in source, f'{name} missing PASS banner literal'


def test_workflow_invokes_every_gate() -> None:
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8')
    for script in GATE_SCRIPTS:
        owner = (REPO_ROOT / '.github/workflows/pr_checks_version.yml').read_text() if script in (
            'check_coverage_ratchet.py', 'check_budget_ratchet.py'
        ) else workflow
        assert f'governance/{script}' in owner, f'{script} not invoked by its required workflow'
    assert 'governance/check_quality_debt.py' in workflow
    assert 'governance/check_quality_debt.py' in workflow
    runner = (GOVERNANCE_DIR / 'check_quality_debt.py').read_text()
    assert 'min_confidence=80' in runner
    assert 'governance/ruff.toml' in runner


def test_no_soft_fail_pathway_in_workflow() -> None:
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8').split('  publish_coverage_comment:')[0]
    assert '|| true' not in workflow
    assert 'continue-on-error' not in workflow
    forbidden_flags = re.compile(r'--warn-only|--no-fail|--soft(-fail)?')
    assert forbidden_flags.search(workflow) is None


def test_scripts_are_self_budgeted() -> None:
    data = json.loads(BUDGET_JSON.read_text(encoding='utf-8'))[BUDGET_SECTION]
    for name in GATE_SCRIPTS:
        key = f'governance/{name}'
        assert key in data, f'{key} missing from budgets.json "modules"'
        assert data[key] <= 120, f'{key} budget {data[key]} exceeds the 120-line self-limit'


def test_ruff_select_includes_new_rules() -> None:
    select = loads_toml((GOVERNANCE_DIR / 'ruff.toml').read_text())['lint']['select']
    for rule in ('C901', 'PLR0912', 'PLR0913', 'PLR0915', 'T201',
                 'FIX001', 'FIX002', 'FIX003', 'FIX004',
                 'ERA001', 'D200', 'D205', 'D415', 'PIE790'):
        assert rule in select, f'ruff select missing {rule}'


def test_budget_ratchet_vacuous_when_base_missing() -> None:
    result = _run('check_budget_ratchet.py', '--base-file', '/dev/null', '--pr-body-file', '/dev/null')
    assert result.returncode == 0
    assert 'BUDGET RATCHET GATE -- PASS' in result.stdout
    assert 'vacuous' in result.stdout.lower()


def test_budget_ratchet_accepts_marker(tmp_path: Path) -> None:
    # Build a self-contained repo layout in tmp_path that actually has a
    # budget raise between base and head. Previous version ran against
    # the real head budget, so the base's `foo.py` key never matched
    # anything in head and the marker logic was never exercised.
    (tmp_path / '.github').mkdir()
    head = {'talos/foo.py': 200}
    base = {'talos/foo.py': 100}
    (tmp_path / '.github' / 'budgets.json').write_text(
        json.dumps({BUDGET_SECTION: head}), encoding='utf-8')
    base_file = tmp_path / 'base.json'
    base_file.write_text(json.dumps(base), encoding='utf-8')
    body_file = tmp_path / 'body.txt'
    body_file.write_text(
        '[budget-raise: talos/foo.py: legitimate growth]\n',
        encoding='utf-8',
    )
    scripts_dir = tmp_path / 'governance'
    scripts_dir.mkdir()
    (scripts_dir / '__init__.py').write_text('', encoding='utf-8')
    import shutil
    shutil.copy2(GOVERNANCE_DIR / '_common.py', scripts_dir / '_common.py')
    shutil.copy2(GOVERNANCE_DIR / 'check_budget_ratchet.py', scripts_dir / 'check_budget_ratchet.py')
    result = subprocess.run(
        [sys.executable, str(scripts_dir / 'check_budget_ratchet.py'),
         '--base-file', str(base_file), '--pr-body-file', str(body_file)],
        check=False, capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'BUDGET RATCHET GATE -- PASS' in result.stdout


def test_no_module_imports_tomllib_unguarded() -> None:
    """Every TOML read goes through `_common.loads_toml`, never a bare import.

    `tomllib` is stdlib only from 3.11. This repository targets 3.12, so a bare
    `import tomllib` passes every gate here and breaks on the first derived
    repository with a lower floor -- which is what happened, across five
    separate surfaces. Nothing in this repository's own CI exercises the
    failing path, so only a scan catches the reintroduction.

    `_common` itself carries the one guarded import, indented inside its `try`,
    which is the fix rather than the defect.
    """
    offenders: list[str] = []
    for path in sorted(GOVERNANCE_DIR.rglob('*.py')):
        for lineno, line in enumerate(
            path.read_text(encoding='utf-8').splitlines(), start=1
        ):
            if line == 'import tomllib':
                offenders.append(f'{path.relative_to(REPO_ROOT)}:{lineno}')
    assert not offenders, (
        'unguarded module-level `import tomllib` -- read TOML through '
        f'`_common.loads_toml` instead: {offenders}'
    )


def test_no_gate_test_shells_out_to_a_bare_python3() -> None:
    """Subprocess tests must invoke `sys.executable`, not a bare `python3`.

    A bare `python3` runs the system interpreter, which below 3.11 has neither
    `tomllib` nor the `tomli` the venv installed. The gate then dies on
    ModuleNotFoundError and the test asserts against the wrong failure.
    """
    offenders: list[str] = []
    for path in sorted((GOVERNANCE_DIR / 'tests').glob('*.py')):
        for lineno, line in enumerate(
            path.read_text(encoding='utf-8').splitlines(), start=1
        ):
            # The subprocess-argument form exactly, so a comment or a pattern
            # literal naming it is not a hit.
            if line.strip() in ("'python3',", '"python3",'):
                offenders.append(f'{path.relative_to(REPO_ROOT)}:{lineno}')
    assert not offenders, f'use sys.executable instead: {offenders}'
