"""Docstring conventions gate: the three mechanizable rules fire and pass."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import types
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> types.ModuleType:
    # governance/ is on sys.path via governance/tests/conftest.py.
    return importlib.import_module('check_docstrings')


def test_forbidden_title_verb_is_flagged() -> None:
    mod = _load()
    assert mod.check_docstring('Calculate the average.') != []
    assert mod.check_docstring('Compute the average.') == []


def test_default_in_description_is_flagged() -> None:
    mod = _load()
    body = 'Compute x.\n\nArgs:\n    p (int): periods (default: 14)'
    assert mod.check_docstring(body) != []
    assert mod.check_docstring('Compute x.\n\nArgs:\n    p (int): periods') == []


def test_note_casing_is_flagged() -> None:
    mod = _load()
    assert mod.check_docstring('Compute x.\n\nNote: be careful.') != []
    assert mod.check_docstring('Compute x.\n\nNOTE: be careful.') == []


def test_find_violations_locates_the_function() -> None:
    mod = _load()
    src = 'def f():\n    """Calculate it."""\n    return 1\n'
    found = mod.find_violations(src)
    assert len(found) == 1
    assert found[0][1] == 'f'


def _pkg(tmp_path: Path, files: dict[str, str], config: dict[str, object]) -> subprocess.CompletedProcess[str]:
    """Run the module-docstring gate over a throwaway package."""
    (tmp_path / '.github').mkdir()
    (tmp_path / 'governance').mkdir()
    (tmp_path / 'pkg').mkdir()
    for name, body in files.items():
        (tmp_path / 'pkg' / name).write_text(body, encoding='utf-8')
    (tmp_path / 'governance.yml').write_text(
        yaml.safe_dump({
            'schema_version': 2,
            'layout': {'package_root': 'pkg', 'excludes': []},
            'gates': {'module_docstrings': config},
        }),
        encoding='utf-8')
    for mod in ('_common.py', '_quality.py', 'check_module_docstrings.py'):
        shutil.copy(REPO_ROOT / 'governance' / mod, tmp_path / 'governance' / mod)
    return subprocess.run(
        [sys.executable, str(tmp_path / 'governance' / 'check_module_docstrings.py')],
        capture_output=True, text=True, cwd=tmp_path, check=False)


def test_exemption_waives_only_the_requirement_to_carry_a_docstring(tmp_path: Path) -> None:
    """An exempt module that carries a docstring still owes the convention.

    The exemptions are documented as waiving the requirement to *have* one,
    and law 6 says the docstring conventions still hold. Skipping the whole
    check would silently waive the one-line rule too.
    """
    result = _pkg(
        tmp_path,
        {
            # exempt by the filename rule, but its docstring spans three lines
            'log_returns.py': '"""Line one.\n\nLine two."""\n\n\ndef log_returns(x):\n    return x\n',
            # exempt by the filename rule and carries none: legitimately silent
            'sma_ratios.py': 'def sma_ratios(x):\n    return x\n',
        },
        {'exempt_filename_matching_single_symbol': True},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'log_returns.py: module docstring spans 3 lines' in result.stderr
    assert 'sma_ratios.py' not in result.stderr


def test_non_boolean_exemption_flag_fails_closed(tmp_path: Path) -> None:
    """The boolean exemption is validated like its two siblings.

    Left unvalidated it was read with an identity test against True, so a
    string or an int silently disabled the exemption instead of blocking.
    """
    result = _pkg(
        tmp_path,
        {'a.py': '"""One."""\n'},
        {'exempt_filename_matching_single_symbol': 'yes'},
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'must be a boolean' in result.stderr
