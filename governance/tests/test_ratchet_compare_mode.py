"""Compare mode: the base-vs-head ratchet that stops a PR raising its own ceiling.

Both escape-hatch gates are ratchets. Their head-only checks assert the counts
sit under the budget; their *compare* checks assert the budget itself was not
loosened, and the scan surface not narrowed, by the same PR the gate is
guarding. Without compare mode a PR can raise every total to whatever its own
code needs and pass.

That mechanism had no test at all. The cost showed up in review: a
`NameError` on `fail_loud_gate.gate()`'s compare path reached CI, because
nothing in the suite called it. These tests therefore drive the real entry
points -- `gate_budget_source`, `gate`, `scan_surface_failures` -- rather than
reimplementing the comparison, so a gate that cannot execute fails here first.
"""
from __future__ import annotations

import importlib
import json
import types
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _mod(name: str) -> types.ModuleType:
    # governance/ is on sys.path via governance/tests/conftest.py.
    return importlib.import_module(name)


common = _mod('_common')
typing_gate = _mod('typing_gate')
fail_loud = _mod('fail_loud_gate')


def _budget_file(tmp_path: Path, section: str, payload: dict[str, object]) -> str:
    path = tmp_path / f'base_{section}.json'
    path.write_text(json.dumps({section: payload}), encoding='utf-8')
    return str(path)


def _config_file(tmp_path: Path, layout: dict[str, object]) -> str:
    path = tmp_path / 'base_governance.yml'
    path.write_text(yaml.safe_dump({'layout': layout}), encoding='utf-8')
    return str(path)


def _head_layout() -> dict[str, object]:
    return dict(common.section('layout', 'TEST'))


# --- the budget itself cannot be loosened -------------------------------

def test_typing_compare_rejects_a_raised_total(tmp_path: Path) -> None:
    """Raising a pattern total above the base ref is the bypass this blocks."""
    base = {
        'patterns': {'noqa': {'pattern': r'#\s*noqa', 'total': 0}},
        'any_references': {'total': 0},
        'pyright_errors': {'total': 0},
    }
    head = {
        'patterns': {'noqa': {'pattern': r'#\s*noqa', 'total': 5}},
        'any_references': {'total': 0},
        'pyright_errors': {'total': 0},
    }
    failures = typing_gate.gate_budget_source(
        _budget_file(tmp_path, 'typing', base),
        _config_file(tmp_path, _head_layout()),
        False,
        head,
    )
    assert failures, 'a raised total must be reported'
    assert any('noqa' in f for f in failures), failures


def test_typing_compare_accepts_a_lowered_total(tmp_path: Path) -> None:
    """Lowering is the ratchet working, and must not be reported."""
    base = {
        'patterns': {'noqa': {'pattern': r'#\s*noqa', 'total': 5}},
        'any_references': {'total': 3},
        'pyright_errors': {'total': 2},
    }
    head = {
        'patterns': {'noqa': {'pattern': r'#\s*noqa', 'total': 0}},
        'any_references': {'total': 0},
        'pyright_errors': {'total': 0},
    }
    failures = typing_gate.gate_budget_source(
        _budget_file(tmp_path, 'typing', base),
        _config_file(tmp_path, _head_layout()),
        False,
        head,
    )
    assert failures == [], failures


def test_typing_compare_rejects_a_raised_section_total(tmp_path: Path) -> None:
    """`any_references` and `pyright_errors` are guarded separately from patterns.

    They live in their own budget sections with their own comparison, so a
    test that only covers the per-pattern totals leaves this half untested --
    which is how the first mutation run found this gap.
    """
    base = {
        'patterns': {'noqa': {'pattern': r'#\s*noqa', 'total': 0}},
        'any_references': {'total': 0},
        'pyright_errors': {'total': 0},
    }
    head = {
        'patterns': {'noqa': {'pattern': r'#\s*noqa', 'total': 0}},
        'any_references': {'total': 7},
        'pyright_errors': {'total': 0},
    }
    failures = typing_gate.gate_budget_source(
        _budget_file(tmp_path, 'typing', base),
        _config_file(tmp_path, _head_layout()),
        False,
        head,
    )
    assert any('any_references' in f for f in failures), failures


def test_typing_compare_rejects_a_deleted_pattern(tmp_path: Path) -> None:
    """Deleting a pattern key is the same bypass as raising its total."""
    base = {
        'patterns': {
            'noqa': {'pattern': r'#\s*noqa', 'total': 0},
            'type_ignore': {'pattern': r'#\s*type:\s*ignore', 'total': 0},
        },
        'any_references': {'total': 0},
        'pyright_errors': {'total': 0},
    }
    head = {
        'patterns': {'noqa': {'pattern': r'#\s*noqa', 'total': 0}},
        'any_references': {'total': 0},
        'pyright_errors': {'total': 0},
    }
    failures = typing_gate.gate_budget_source(
        _budget_file(tmp_path, 'typing', base),
        _config_file(tmp_path, _head_layout()),
        False,
        head,
    )
    assert any('type_ignore' in f for f in failures), failures


def test_fail_loud_compare_mode_executes_and_rejects_a_raise(tmp_path: Path) -> None:
    """The compare path runs, and reports a raised category total.

    Executing at all is half the assertion: this path shipped a `NameError`
    because no test had ever called it.
    """
    base = {'categories': {'bare_except': {'total': 0}}}
    head = {'categories': {'bare_except': {'total': 4}}}
    failures = fail_loud.gate(head, base, _config_file(tmp_path, _head_layout()))
    assert failures, 'a raised category total must be reported'
    assert any('bare_except' in f for f in failures), failures


def test_fail_loud_compare_accepts_an_unchanged_budget(tmp_path: Path) -> None:
    """A PR that touches neither the budget nor the surface reports nothing."""
    budget = json.loads((REPO_ROOT / '.github/budgets.json').read_text())['fail_loud']
    failures = fail_loud.gate(budget, budget, _config_file(tmp_path, _head_layout()))
    assert failures == [], failures


# --- the scan surface cannot be narrowed --------------------------------

def test_narrowed_package_root_is_rejected(tmp_path: Path) -> None:
    """Pointing the gate at a different subtree changes what the count means."""
    layout = _head_layout() | {'package_root': 'some_other_package'}
    failures = common.scan_surface_failures(_config_file(tmp_path, layout), 'TEST')
    assert any('package_root' in f for f in failures), failures


def test_added_exclude_is_rejected(tmp_path: Path) -> None:
    """A new exclude hides files from the ratchet, so head cannot add one."""
    base_layout = dict(_head_layout())
    base_layout['excludes'] = [
        e for e in base_layout.get('excludes', []) if e != 'dist'
    ]
    failures = common.scan_surface_failures(_config_file(tmp_path, base_layout), 'TEST')
    assert any('excludes' in f for f in failures), failures
    assert any('dist' in f for f in failures), failures


def test_removed_exclude_is_accepted(tmp_path: Path) -> None:
    """Widening the scan surface is a tightening, and must not be reported."""
    wider = dict(_head_layout())
    wider['excludes'] = [*wider.get('excludes', []), 'an_extra_base_only_exclude']
    failures = common.scan_surface_failures(_config_file(tmp_path, wider), 'TEST')
    assert failures == [], failures


def test_unchanged_surface_is_accepted(tmp_path: Path) -> None:
    """The repository's own config compared against itself reports nothing."""
    failures = common.scan_surface_failures(_config_file(tmp_path, _head_layout()), 'TEST')
    assert failures == [], failures


def test_missing_base_config_blocks_rather_than_passing(tmp_path: Path) -> None:
    """An unreadable base surface means the ratchet cannot be trusted.

    Returning no failures here would let deleting `governance.yml` from the
    base ref silently disable the surface comparison.
    """
    failures = common.scan_surface_failures(str(tmp_path / 'absent.yml'), 'TEST')
    assert failures, 'a missing base config must be reported'
    assert any('not found' in f for f in failures), failures


def _config_with_gates(tmp_path: Path, layout: dict[str, object],
                       gates: dict[str, object]) -> str:
    """A base config carrying both halves of the scan surface."""
    path = tmp_path / 'base_with_gates.yml'
    path.write_text(yaml.safe_dump({'layout': layout, 'gates': gates}), encoding='utf-8')
    return str(path)


def test_added_per_gate_exclude_is_rejected(tmp_path: Path) -> None:
    """A `gates.typing.excludes` entry narrows the surface just as `layout` does.

    The gates resolve their tree through `layout_excludes`, which merges the
    per-gate list into the repo-wide one. Comparing only the repo-wide half
    left this as an unguarded lever: four lines of config hid a file from both
    ratchets while both still reported PASS.
    """
    base = _config_with_gates(tmp_path, _head_layout(), {'typing': {'enabled': True}})
    head_gates = dict(common.section('gates', 'TEST'))
    head_gates['typing'] = {**head_gates.get('typing', {}), 'excludes': ['sneaky.py']}
    import unittest.mock as mock
    with mock.patch.object(common, 'config', return_value={
        'layout': _head_layout(), 'gates': head_gates,
    }):
        failures = common.scan_surface_failures(base, 'TEST', 'typing')
    assert failures, 'a per-gate exclude must be reported'
    assert any('sneaky.py' in f for f in failures), failures


def test_per_gate_exclude_unchanged_is_accepted(tmp_path: Path) -> None:
    """The repository compared against itself reports nothing, gate included."""
    base = _config_with_gates(
        tmp_path, _head_layout(), dict(common.section('gates', 'TEST'))
    )
    assert common.scan_surface_failures(base, 'TEST', 'typing') == []
    assert common.scan_surface_failures(base, 'TEST', 'fail_loud') == []
