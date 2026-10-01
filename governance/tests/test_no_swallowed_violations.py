"""Pin the AST gate's detection across the three honesty-violation bypass shapes.

The previous shell-grep implementation only caught the bare-name form
(`except HonestyViolation`). This test pins that the AST replacement
also catches the dotted form and the aliased form — and that it does
NOT flag innocent excepts (false-positive guard).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / 'governance' / 'check_no_swallowed_violations.py'


def _load_gate():
    spec = importlib.util.spec_from_file_location('cnsv_gate', GATE_PATH)
    if spec is None or spec.loader is None:
        msg = f'failed to load gate module from {GATE_PATH}'
        raise RuntimeError(msg)
    m = importlib.util.module_from_spec(spec)
    sys.modules['cnsv_gate'] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope='module')
def gate():
    return _load_gate()


def _write(tmp_path: Path, src: str) -> Path:
    path = tmp_path / 'sample.py'
    path.write_text(src, encoding='utf-8')
    return path


def test_detects_bare_name(tmp_path: Path, gate) -> None:
    src = (
        'from talos.exceptions import HonestyViolation\n'
        'def f():\n'
        '    try:\n'
        '        do_thing()\n'
        '    except HonestyViolation:\n'
        '        pass\n'
    )
    findings = gate.check_file(_write(tmp_path, src))
    assert findings == [(5, 'HonestyViolation')], findings


def test_detects_subclass(tmp_path: Path, gate) -> None:
    src = (
        'from talos.exceptions import LookAheadViolation\n'
        'def f():\n'
        '    try:\n'
        '        do_thing()\n'
        '    except LookAheadViolation:\n'
        '        pass\n'
    )
    findings = gate.check_file(_write(tmp_path, src))
    assert findings == [(5, 'LookAheadViolation')], findings


def test_detects_dotted_form(tmp_path: Path, gate) -> None:
    # The shell-grep predecessor missed this because its regex only
    # anchored on bare exception names.
    src = (
        'import talos.exceptions\n'
        'def f():\n'
        '    try:\n'
        '        do_thing()\n'
        '    except talos.exceptions.HonestyViolation:\n'
        '        pass\n'
    )
    findings = gate.check_file(_write(tmp_path, src))
    assert findings == [(5, 'HonestyViolation')], findings


def test_detects_aliased_import(tmp_path: Path, gate) -> None:
    # The shell-grep predecessor was bypassable via aliasing the import.
    # The AST gate resolves through the alias map so `except HV` flags
    # the same as `except HonestyViolation`.
    src = (
        'from talos.exceptions import HonestyViolation as HV\n'
        'def f():\n'
        '    try:\n'
        '        do_thing()\n'
        '    except HV:\n'
        '        pass\n'
    )
    findings = gate.check_file(_write(tmp_path, src))
    assert findings == [(5, 'HonestyViolation')], findings


def test_detects_in_tuple(tmp_path: Path, gate) -> None:
    src = (
        'from talos.exceptions import HonestyViolation\n'
        'def f():\n'
        '    try:\n'
        '        do_thing()\n'
        '    except (ValueError, HonestyViolation):\n'
        '        pass\n'
    )
    findings = gate.check_file(_write(tmp_path, src))
    assert findings == [(5, 'HonestyViolation')], findings


def test_innocent_except_not_flagged(tmp_path: Path, gate) -> None:
    src = (
        'def f():\n'
        '    try:\n'
        '        do_thing()\n'
        '    except ValueError:\n'
        '        pass\n'
    )
    assert gate.check_file(_write(tmp_path, src)) == []


def test_bare_except_not_flagged(tmp_path: Path, gate) -> None:
    # Bare `except:` is forbidden by a DIFFERENT gate (fail_loud_gate);
    # this gate only fires on caught HonestyViolation names.
    src = (
        'def f():\n'
        '    try:\n'
        '        do_thing()\n'
        '    except:\n'
        '        pass\n'
    )
    assert gate.check_file(_write(tmp_path, src)) == []


def test_innocent_aliased_import_not_flagged(tmp_path: Path, gate) -> None:
    # False-positive guard: aliasing an UNRELATED exception must not
    # flag. The alias resolver is for tracking import aliases, not for
    # blanket-flagging anything that happens to have an asname.
    src = (
        'from talos.exceptions import HonestyViolation as HV\n'
        'def f():\n'
        '    try:\n'
        '        do_thing()\n'
        '    except ValueError:\n'
        '        pass\n'
    )
    assert gate.check_file(_write(tmp_path, src)) == []


def test_live_repo_passes(gate) -> None:
    # Sanity check on the actual codebase: the gate must currently pass.
    # If this ever fails, somebody added a swallowed honesty violation.
    findings_total = 0
    for target_name in ('talos', 'tests'):
        target = REPO_ROOT / target_name
        if not target.exists():
            continue
        for py_file in target.rglob('*.py'):
            findings_total += len(gate.check_file(py_file))
    assert findings_total == 0, f'live repo has {findings_total} swallowed violations'


def test_function_scope_alias_does_not_mask_module_alias(tmp_path: Path, gate) -> None:
    """Adversarial bypass: function-local rebinding must not mask another scope.

    Pre-fix the gate built ONE alias map for the entire file, so a
    function-local `from decimal import Decimal as HonestyViolation`
    overwrote the module-level binding and made the gate miss the
    real swallow in a sibling function. Per-scope tracking fixes it.
    """
    src = (
        'from talos.exceptions import HonestyViolation\n'
        '\n'
        'def real_swallow():\n'
        '    try:\n'
        '        do_thing()\n'
        '    except HonestyViolation:  # this MUST be flagged\n'
        '        pass\n'
        '\n'
        'def adversarial_rebind():\n'
        '    from decimal import Decimal as HonestyViolation  # noqa: F811\n'
        '    try:\n'
        '        return Decimal(\'1\')  # noqa: F821\n'
        '    except HonestyViolation:  # actually catches Decimal\n'
        '        pass\n'
    )
    findings = gate.check_file(_write(tmp_path, src))
    flagged_lines = [lineno for lineno, _ in findings]
    assert 6 in flagged_lines, (
        f'expected gate to flag the real swallow at line 6 despite a '
        f'function-local rebinding in a sibling function; got {findings}'
    )


def test_function_local_alias_resolves_in_same_function(tmp_path: Path, gate) -> None:
    """Function-local `import X as HV` + `except HV` must still flag.

    Per-scope aliases must search innermost-out and detect a swallow
    when the alias is defined in the same scope as the `except`.
    """
    src = (
        'def f():\n'
        '    from talos.exceptions import HonestyViolation as HV\n'
        '    try:\n'
        '        do_thing()\n'
        '    except HV:\n'
        '        pass\n'
    )
    findings = gate.check_file(_write(tmp_path, src))
    assert findings == [(5, 'HonestyViolation')], findings


def test_class_body_alias_does_not_mask_method_swallow(tmp_path: Path, gate) -> None:
    """Class-body alias must not bleed into method-scope name resolution.

    Python methods do NOT close over class-body bindings as bare
    names; a class-body `HonestyViolation = SomethingElse` does not
    affect how `except HonestyViolation` resolves inside a method —
    the method sees the module-level binding. The gate must mirror
    that semantic so a class-body alias can't fool it.
    """
    src = (
        'from talos.exceptions import HonestyViolation\n'
        '\n'
        'class C:\n'
        '    from decimal import Decimal as HonestyViolation  # noqa: F811\n'
        '    def method(self):\n'
        '        try:\n'
        '            do_thing()\n'
        '        except HonestyViolation:  # at runtime catches the real one\n'
        '            pass\n'
    )
    findings = gate.check_file(_write(tmp_path, src))
    flagged_lines = [lineno for lineno, _ in findings]
    assert 8 in flagged_lines, (
        f'expected gate to flag the method-level swallow at line 8 '
        f'despite a class-body alias rebinding; got {findings}. The '
        f'gate must skip class scopes when resolving method names.'
    )
