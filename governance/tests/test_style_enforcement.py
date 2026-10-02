"""Verify the required style command rejects real source defects."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from _common import loads_toml

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ['talos', 'governance', 'tests', 'tools', 'scripts', 'examples']
DOCSTRING_RULES = {'D200', 'D205', 'D415'}


def _style_command() -> list[str]:
    workflow = yaml.safe_load((ROOT / '.github/workflows/pr_checks_lint.yml').read_text())
    steps = workflow['jobs']['pr_checks_lint']['steps']
    step = next(item for item in steps if item['name'] == 'Enforce selected Python style with zero findings')
    return shlex.split(step['run'])


@pytest.fixture
def style_project(tmp_path: Path) -> Path:
    shutil.copyfile(ROOT / 'pyproject.toml', tmp_path / 'pyproject.toml')
    for scope in SCOPES:
        (tmp_path / scope).mkdir()
    shutil.copyfile(ROOT / 'governance/ruff.toml', tmp_path / 'governance/ruff.toml')
    return tmp_path


def _run_style(root: Path) -> subprocess.CompletedProcess[str]:
    command = _style_command()
    return subprocess.run(
        [sys.executable, *command[1:]], cwd=root, text=True, capture_output=True, check=False,
        env={**os.environ, 'RUFF_OUTPUT_FORMAT': 'json'})


def _codes(result: subprocess.CompletedProcess[str]) -> set[str]:
    return {finding['code'] for finding in json.loads(result.stdout)}


def test_style_workflow_keeps_the_required_context_and_exact_scope() -> None:
    workflow = yaml.safe_load((ROOT / '.github/workflows/pr_checks_lint.yml').read_text())
    assert set(workflow['jobs']) == {'pr_checks_lint'}
    assert workflow['jobs']['pr_checks_lint']['name'] == 'pr_checks_lint'
    steps = workflow['jobs']['pr_checks_lint']['steps']
    matches = [item for item in steps if item['name'] == 'Enforce selected Python style with zero findings']
    assert len(matches) == 1
    assert 'if' not in matches[0]
    assert 'continue-on-error' not in matches[0]
    assert _style_command() == [
        '.venv-lint/bin/python', '-m', 'ruff', 'check', '--config', 'governance/ruff.toml', '--preview',
        '--select', 'E,W,I,D200,D205,D415,RUF022', '--ignore', 'E501', *SCOPES]
    assert any('governance/check_quality_debt.py' in item.get('run', '') for item in steps)


def test_style_uses_only_existing_rule_exceptions() -> None:
    config = loads_toml((ROOT / 'pyproject.toml').read_text())['tool']['ruff']
    assert config['lint']['ignore'] == ['E501']
    exceptions = {
        pattern: {
            rule for rule in rules
            if re.fullmatch(r'(E|W|I)\d*', rule) or rule in DOCSTRING_RULES or rule == 'RUF022'
        }
        for pattern, rules in config['lint']['per-file-ignores'].items()
    }
    assert exceptions == {
        'tests/**/*.py': DOCSTRING_RULES,
        'governance/*.py': DOCSTRING_RULES,
        'scripts/*.py': DOCSTRING_RULES,
        'governance/tests/**/*.py': DOCSTRING_RULES,
    }
    assert config['exclude'] == [
        '.git', '__pycache__', 'build', 'dist', 'governance/tests/fixtures']


@pytest.mark.parametrize('rule,bad,good', [
    ('E225', 'value=1\n', 'value = 1\n'),
    ('E702', 'first = 1; second = 2\n', 'first = 1\nsecond = 2\n'),
    ('I001', 'import sys\nimport os\n', 'import os\nimport sys\n'),
    ('D200', '"""\nDescribe behavior.\n"""\n', '"""Describe behavior."""\n'),
    ('D205', '"""Describe behavior.\nMore detail.\n"""\n', '"""Describe behavior.\n\nMore detail.\n"""\n'),
    ('D415', '"""Describe behavior"""\n', '"""Describe behavior."""\n'),
    ('RUF022', "__all__ = ['zebra', 'alpha']\n", "__all__ = ['alpha', 'zebra']\n"),
])
def test_style_rejects_defects_and_accepts_their_corrections(
    style_project: Path, rule: str, bad: str, good: str,
) -> None:
    target = style_project / 'talos/probe.py'
    target.write_text(bad)
    rejected = _run_style(style_project)
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert rule in _codes(rejected), rejected.stdout
    target.write_text(good)
    accepted = _run_style(style_project)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr


@pytest.mark.parametrize('scope', SCOPES)
def test_style_checks_whitespace_in_every_declared_scope(style_project: Path, scope: str) -> None:
    target = style_project / scope / 'probe.py'
    target.write_text('value=1\n')
    rejected = _run_style(style_project)
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert _codes(rejected) == {'E225'}, rejected.stdout
    target.write_text('value = 1\n')
    accepted = _run_style(style_project)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr


@pytest.mark.parametrize('path,docstring_ignored', [
    ('talos/probe.py', False),
    ('governance/probe.py', True),
    ('governance/tests/test_probe.py', True),
    ('tests/test_probe.py', True),
    ('scripts/probe.py', True),
    ('tools/probe.py', False),
    ('examples/probe.py', False),
])
def test_existing_docstring_boundaries_do_not_disable_whitespace(
    style_project: Path, path: str, docstring_ignored: bool,
) -> None:
    target = style_project / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('"""\nDescribe behavior.\n"""\nvalue=1\n')
    rejected = _run_style(style_project)
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert _codes(rejected) == ({'E225'} if docstring_ignored else {'D200', 'E225'}), rejected.stdout
    target.write_text('"""Describe behavior."""\nvalue = 1\n')
    accepted = _run_style(style_project)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr


def test_existing_line_length_exclusion_remains(style_project: Path) -> None:
    (style_project / 'talos/probe.py').write_text("text = '" + 'x' * 120 + "'\n")
    result = _run_style(style_project)
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_repository_has_zero_selected_style_findings() -> None:
    result = _run_style(ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def _compiler_command() -> list[str]:
    workflow = yaml.safe_load((ROOT / '.github/workflows/pr_checks_lint.yml').read_text())
    steps = workflow['jobs']['pr_checks_lint']['steps']
    step = next(item for item in steps if item['name'] == 'Reject Python compiler warnings')
    return shlex.split(step['run'])


def test_compiler_warning_gate_keeps_all_source_scopes() -> None:
    workflow = yaml.safe_load((ROOT / '.github/workflows/pr_checks_lint.yml').read_text())
    steps = workflow['jobs']['pr_checks_lint']['steps']
    matches = [item for item in steps if item['name'] == 'Reject Python compiler warnings']
    assert len(matches) == 1
    assert 'if' not in matches[0]
    assert 'continue-on-error' not in matches[0]
    assert _compiler_command() == [
        '.venv-lint/bin/python', '-Werror', '-m', 'compileall', '-q', '-f', *SCOPES]


@pytest.mark.parametrize('scope', SCOPES)
def test_compiler_warning_is_rejected_before_execution(style_project: Path, scope: str) -> None:
    target = style_project / scope / 'probe.py'
    target.write_text("pattern = '\\d'\n")
    command = [sys.executable, *_compiler_command()[1:]]
    rejected = subprocess.run(command, cwd=style_project, capture_output=True, text=True, check=False)
    assert rejected.returncode == 1
    assert 'invalid escape sequence' in rejected.stdout + rejected.stderr
    target.write_text("pattern = r'\\d'\n")
    accepted = subprocess.run(command, cwd=style_project, capture_output=True, text=True, check=False)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
