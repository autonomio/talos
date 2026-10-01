"""CodeQL fallback: bootstrap can mechanically remove CodeQL and keep the bijection."""
from __future__ import annotations

import importlib
import json
import re
import shutil
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_WORKFLOW = REPO_ROOT / '.github/workflows/bootstrap_repository.yml'

_LAW_LINE = re.compile(r'^\d+\.\s')
_ANNOTATION = re.compile(r'\*\((?P<a>.+)\)\*\s*$')

# When the repo has already had CodeQL removed (e.g. a private bootstrapped
# repo), disable_codeql is a no-op; the removal tests have nothing to assert.
_CODEQL_PRESENT = 'PR Checks CodeQL (python)' in (REPO_ROOT / 'CLAUDE.md').read_text(encoding='utf-8')
_skip_if_no_codeql = pytest.mark.skipif(
    not _CODEQL_PRESENT,
    reason='CodeQL already removed from this repo',
)


def _bootstrap() -> types.ModuleType:
    # governance/ is on sys.path via governance/tests/conftest.py.
    return importlib.import_module('bootstrap_repository')


def _laws_section(text: str) -> list[str]:
    out: list[str] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith('## '):
            in_section = line.strip() == '## The laws'
            continue
        if in_section:
            out.append(line)
    return out


def _copy_template(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    shutil.copytree(
        REPO_ROOT,
        repo,
        ignore=shutil.ignore_patterns(
            '.git', '.venv*', '__pycache__', '.pytest_cache', '.ruff_cache',
            'node_modules', 'build', '.generated', '.docusaurus', 'test-results',
        ),
    )
    return repo


@_skip_if_no_codeql
def test_disable_codeql_removes_law_ruleset_and_workflow(tmp_path: Path) -> None:
    repo = _copy_template(tmp_path)
    changed = _bootstrap().disable_codeql(repo)

    assert changed >= 3
    assert not (repo / '.github/workflows/pr_checks_codeql.yml').exists()

    ruleset = json.loads((repo / '.github/rulesets/master.json').read_text(encoding='utf-8'))
    checks = next(r for r in ruleset['rules'] if r['type'] == 'required_status_checks')
    contexts = {c['context'] for c in checks['parameters']['required_status_checks']}
    assert 'PR Checks CodeQL (python)' not in contexts
    assert not any(rule['type'] == 'code_scanning' for rule in ruleset['rules'])

    laws = (repo / 'CLAUDE.md').read_text(encoding='utf-8')
    assert 'CodeQL reports' not in laws
    assert 'PR Checks CodeQL (python)' not in laws
    assert 'Ten laws. Nine are workflow gates on every PR; the tenth' in laws

    # governance.yml is the contract anchor test_governance_config pins the
    # ruleset to; if disable_codeql leaves CodeQL here, a private bootstrap PR
    # fails that check. Guard the regression where the template's own CI (which
    # always has CodeQL) cannot otherwise see it.
    config = (repo / 'governance.yml').read_text(encoding='utf-8')
    assert 'PR Checks CodeQL (python)' not in config


@_skip_if_no_codeql
def test_disable_codeql_renumbers_laws_sequentially(tmp_path: Path) -> None:
    repo = _copy_template(tmp_path)
    _bootstrap().disable_codeql(repo)
    section = '\n'.join(_laws_section((repo / 'CLAUDE.md').read_text(encoding='utf-8')))
    numbers = [int(m.group(1)) for m in re.finditer(r'^(\d+)\.\s', section, re.MULTILINE)]
    assert numbers == list(range(1, len(numbers) + 1)), numbers


@_skip_if_no_codeql
def test_disable_codeql_preserves_bijection(tmp_path: Path) -> None:
    repo = _copy_template(tmp_path)
    _bootstrap().disable_codeql(repo)

    annotations: list[str] = []
    for line in _laws_section((repo / 'CLAUDE.md').read_text(encoding='utf-8')):
        if _LAW_LINE.match(line):
            m = _ANNOTATION.search(line)
            assert m is not None, line
            annotations.append(m.group('a').strip())
    law_contexts = {a for a in annotations if a != 'branch protection, server-side'}

    ruleset = json.loads((repo / '.github/rulesets/master.json').read_text(encoding='utf-8'))
    checks = next(r for r in ruleset['rules'] if r['type'] == 'required_status_checks')
    contexts = {c['context'] for c in checks['parameters']['required_status_checks']}
    assert law_contexts == contexts


def test_bootstrap_workflow_keeps_public_talos_codeql_active() -> None:
    wf = BOOTSTRAP_WORKFLOW.read_text()
    assert '--codeql unsupported' not in wf
    assert (REPO_ROOT / '.github/workflows/pr_checks_codeql.yml').is_file()
    assert 'PR Checks CodeQL (python)' in (REPO_ROOT / 'CLAUDE.md').read_text()


@_skip_if_no_codeql
def test_disable_codeql_preserves_other_scanner_requirements(tmp_path: Path) -> None:
    repo = _copy_template(tmp_path)
    path = repo / '.github/rulesets/master.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    scanning = next(rule for rule in payload['rules'] if rule['type'] == 'code_scanning')
    other_tool = {
        'tool': 'OtherScanner',
        'security_alerts_threshold': 'all',
        'alerts_threshold': 'all',
    }
    scanning['parameters']['code_scanning_tools'].append(other_tool)
    path.write_text(json.dumps(payload), encoding='utf-8')

    _bootstrap().disable_codeql(repo)

    result = json.loads(path.read_text(encoding='utf-8'))
    remaining = next(rule for rule in result['rules'] if rule['type'] == 'code_scanning')
    assert remaining['parameters']['code_scanning_tools'] == [other_tool]
    assert 'PR Checks CodeQL (python)' not in (repo / 'CLAUDE.md').read_text(encoding='utf-8')
