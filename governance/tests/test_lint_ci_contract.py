from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

from _common import loads_toml

REPO_ROOT = Path(__file__).resolve().parents[2]
LINT_WORKFLOW: Final[Path] = REPO_ROOT / '.github/workflows/pr_checks_lint.yml'
RULESET_WORKFLOW: Final[Path] = REPO_ROOT / '.github/workflows/pr_checks_ruleset.yml'
RULESET_SNAPSHOT: Final[Path] = REPO_ROOT / '.github/rulesets/master.json'
DEV_ENV_IN: Final[Path] = REPO_ROOT / 'requirements/ci/dev-env.in'
DEV_ENV_TXT: Final[Path] = REPO_ROOT / 'requirements/ci/dev-env.txt'
BAD_FIXTURE: Final[Path] = REPO_ROOT / 'governance/tests/fixtures/lint/bad_imports.py'
def _pinned_dev_tool(package: str) -> str:
    """The version `pyproject.toml` pins for one dev tool.

    Read rather than restated: a literal here was a sixth place a bump had to
    find, and the one most easily missed because nothing installs from it.
    """
    pyproject = loads_toml((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    dev = pyproject['project']['optional-dependencies']['dev']
    pins = [e.split('==', 1)[1] for e in dev if e.startswith(f'{package}==')]
    assert len(pins) == 1, f'{package} must be pinned exactly once, got {pins}'
    return pins[0]


RUFF_VERSION: Final[str] = _pinned_dev_tool('ruff')
EXPECTED_RUFF_POLICY: Final[dict[str, object]] = {
    "exclude": [
        ".git",
        "__pycache__",
        "build",
        "dist",
        "governance/tests/fixtures"
    ],
    "select": [
        "E",
        "F",
        "I",
        "UP",
        "RUF",
        "BLE",
        "ANN",
        "C901",
        "PLR0912",
        "PLR0913",
        "PLR0915",
        "T201",
        "FIX001",
        "FIX002",
        "FIX003",
        "FIX004",
        "ERA001",
        "D200",
        "D205",
        "D415",
        "PIE790"
    ],
    "ignore": [
        "E501"
    ],
    "per-file-ignores": {
        "tests/**/*.py": [
            "S101",
            "ANN",
            "BLE001",
            "PLR0912",
            "PLR0913",
            "PLR0915",
            "D200",
            "D205",
            "D415"
        ],
        "governance/*.py": [
            "C901",
            "PLR0912",
            "PLR0913",
            "PLR0915",
            "T201",
            "FIX001",
            "FIX002",
            "FIX003",
            "FIX004",
            "ERA001",
            "D200",
            "D205",
            "D415"
        ],
        "scripts/*.py": [
            "C901",
            "PLR0912",
            "PLR0913",
            "PLR0915",
            "T201",
            "D200",
            "D205",
            "D415"
        ],
        "governance/tests/**/*.py": [
            "S101",
            "ANN",
            "BLE001",
            "PLR0912",
            "PLR0913",
            "PLR0915",
            "D200",
            "D205",
            "D415"
        ]
    }
}


def _required_status_contexts() -> list[str]:
    payload = json.loads(RULESET_SNAPSHOT.read_text(encoding='utf-8'))
    for rule in payload['rules']:
        if rule['type'] == 'required_status_checks':
            checks = rule['parameters']['required_status_checks']
            return [entry['context'] for entry in checks]
    raise AssertionError('required_status_checks rule missing from ruleset snapshot')


def _run_ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-m', 'ruff', *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_pr_checks_lint_workflow_exists() -> None:
    assert LINT_WORKFLOW.exists()


def test_ruleset_snapshot_requires_pr_checks_lint() -> None:
    assert 'pr_checks_lint' in _required_status_contexts()


def test_pr_checks_lint_runs_pinned_ruff_on_tools_and_tests_tools() -> None:
    # Name kept for slice #11 Tests-table backward compatibility. The
    # assertions inside now cover the broader post-#11 surface
    # (ruff + every gate script + no-soft-fail). A future slice may
    # split or rename this; doing so requires updating the slice body's
    # Tests table in lockstep.
    workflow = LINT_WORKFLOW.read_text(encoding='utf-8')

    assert '--require-hashes -r requirements/ci/dev-env.txt' in workflow
    assert '--require-hashes -r requirements/ci/runtime-env.txt' in workflow
    assert '--require-hashes -r requirements/ci/build-tools.txt' in workflow
    assert '--no-build-isolation --no-deps -e .' in workflow
    assert 'id: package' in workflow
    assert "yaml.safe_load(Path('governance.yml').read_text())" in workflow
    assert 'governance/check_quality_debt.py' in workflow
    assert '--source="${{ steps.package.outputs.coverage_source }}"' in workflow
    assert '-m pytest tests/test_*.py governance/tests/ -q' in workflow
    assert 'continue-on-error' not in workflow
    # Hard-mechanical gate surfaces from slice #11 — each invocation
    # must appear verbatim somewhere in the workflow.
    assert 'governance/check_module_budgets.py' in workflow
    assert 'governance/check_module_docstrings.py' in workflow
    assert 'governance/check_file_size_balance.py' in workflow
    assert 'governance/check_test_code_ratio.py' in workflow
    assert 'governance/check_coverage_floor.py' in workflow
    assert 'governance/check_coverage_ratchet.py' in workflow
    assert 'governance/check_dependency_vulnerabilities.py' in workflow
    assert 'governance/check_budget_ratchet.py' in workflow
    assert 'uses: actions/setup-node@249970729cb0ef3589644e2896645e5dc5ba9c38  # v6.5.0' in workflow
    assert 'npm --prefix docs-site ci' in workflow
    assert 'node docs-site/node_modules/playwright/cli.js install --with-deps chromium' in workflow
    assert 'npm --prefix docs-site run security:audit' in workflow
    assert 'npm --prefix docs-site run check' in workflow
    assert '--base-ref' in workflow
    # No soft-fail pathway: no `|| true`, no continue-on-error on any step.
    assert '|| true' not in workflow


def test_pr_checks_ruleset_runs_test_lint_ci_contract() -> None:
    workflow = RULESET_WORKFLOW.read_text(encoding='utf-8')

    assert '--require-hashes -r requirements/ci/dev-env.txt' in workflow
    assert 'governance/tests/test_lint_ci_contract.py' in workflow


def test_pinned_ruff_fails_on_known_bad_fixture() -> None:
    version = _run_ruff('--version')
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == f'ruff {RUFF_VERSION}'

    result = _run_ruff('check', '--config', 'governance/ruff.toml', str(BAD_FIXTURE))

    assert result.returncode == 1
    assert 'bad_imports.py' in f'{result.stdout}\n{result.stderr}'


def test_ruff_pin_is_consistent_across_requirement_sets() -> None:
    # The lint and ruleset venvs install the compiled dev-env set, so
    # the ruff the gates run is whatever dev-env pins; source (.in) and
    # compiled (.txt) must both carry exactly the contract version.
    files = [DEV_ENV_IN, DEV_ENV_TXT]
    pins = sorted({
        pin
        for source in files
        for pin in re.findall(r'^ruff==([0-9.]+)', source.read_text(encoding='utf-8'), re.MULTILINE)
    })

    assert pins == [RUFF_VERSION]


def test_pyproject_ruff_policy_contract() -> None:
    data = loads_toml((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    strict = loads_toml((REPO_ROOT / 'governance/ruff.toml').read_text())
    ruff = {**data['tool']['ruff'], 'lint': strict['lint']}
    actual_policy = {
        'exclude': ruff.get('exclude'),
        'select': ruff['lint'].get('select'),
        'ignore': ruff['lint'].get('ignore'),
        'per-file-ignores': ruff['lint'].get('per-file-ignores'),
    }

    assert actual_policy == EXPECTED_RUFF_POLICY

    result = _run_ruff('check', '--config', 'governance/ruff.toml', 'governance')
    assert result.returncode == 0, result.stdout + result.stderr
