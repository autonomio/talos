"""`--update-budget` regenerates one section without destroying the others.

The budgets used to be one file per gate. Each gate's `--update-budget` read
its file, edited it, and wrote the whole thing back -- correct for a file it
owned alone. When the six files merged into `.github/budgets.json` the read
side was sectioned and the write side was not, so `fail_loud_gate
--update-budget` replaced five ratchets with its own `categories` key, and
`typing_gate --update-budget` crashed on a key that had moved to `layout`.

Neither was caught because no test had ever run either command. These drive
the real CLI against a throwaway copy of the repository, because the defect
was in the wiring rather than in any function a unit test would have called.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_IGNORE = shutil.ignore_patterns(
    '.git', '.venv', '.venv-lint', '.venv-ruleset', '.venv-ruleset-audit',
    '__pycache__', 'node_modules', '.pytest_cache', '.ruff_cache', 'htmlcov',
    'build', '.generated', '.docusaurus', 'test-results',
)
# Every section the merged budgets file carries. Regenerating any one of them
# must leave the rest untouched.
SECTIONS = frozenset({'typing', 'fail_loud', 'modules', 'coverage', 'runtime'})


def _scratch(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    shutil.copytree(REPO_ROOT, repo, ignore=_IGNORE)
    return repo


def _budgets(repo: Path) -> dict[str, object]:
    return json.loads((repo / '.github' / 'budgets.json').read_text(encoding='utf-8'))


def _update(repo: Path, gate: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repo / 'governance' / gate), '--update-budget'],
        capture_output=True, text=True, check=False, cwd=repo,
    )


def test_fail_loud_update_preserves_other_sections(tmp_path: Path) -> None:
    """The command that used to leave `{'categories': ...}` and nothing else."""
    repo = _scratch(tmp_path)
    before = _budgets(repo)
    assert SECTIONS <= set(before), 'fixture must start with every section present'

    result = _update(repo, 'fail_loud_gate.py')
    assert result.returncode == 0, result.stdout + result.stderr

    after = _budgets(repo)
    assert SECTIONS <= set(after), f'sections destroyed: {sorted(SECTIONS - set(after))}'
    for section in SECTIONS - {'fail_loud'}:
        assert after[section] == before[section], f'{section} was modified'


def test_typing_update_runs_and_preserves_other_sections(tmp_path: Path) -> None:
    """The command that used to raise KeyError before writing anything."""
    repo = _scratch(tmp_path)
    before = _budgets(repo)

    result = _update(repo, 'typing_gate.py')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'KeyError' not in result.stderr

    after = _budgets(repo)
    assert SECTIONS <= set(after), f'sections destroyed: {sorted(SECTIONS - set(after))}'
    for section in SECTIONS - {'typing'}:
        assert after[section] == before[section], f'{section} was modified'


def test_both_updates_in_sequence_keep_every_section(tmp_path: Path) -> None:
    """Running one after the other is the realistic maintenance sequence.

    Each command alone could preserve the file while the pair still lost a
    section, if the second read a stale copy.
    """
    repo = _scratch(tmp_path)
    assert _update(repo, 'fail_loud_gate.py').returncode == 0
    assert _update(repo, 'typing_gate.py').returncode == 0

    after = _budgets(repo)
    assert SECTIONS <= set(after), f'sections destroyed: {sorted(SECTIONS - set(after))}'
    assert after['modules'], 'the module budgets must survive both regenerations'


def test_update_writes_the_section_it_owns(tmp_path: Path) -> None:
    """Preservation must not be achieved by writing nothing at all."""
    repo = _scratch(tmp_path)
    path = repo / '.github' / 'budgets.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data['fail_loud'] = {'categories': {'bare_except': {'total': 999}}}
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

    assert _update(repo, 'fail_loud_gate.py').returncode == 0
    rewritten = _budgets(repo)['fail_loud']
    assert isinstance(rewritten, dict)
    categories = rewritten['categories']
    assert isinstance(categories, dict)
    assert categories['bare_except']['total'] == 0, 'the section was not regenerated'
