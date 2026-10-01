"""Test fallback gate: try/except in tests is detected; pytest.raises passes."""
from __future__ import annotations

import importlib
import types


def _load() -> types.ModuleType:
    # governance/ is on sys.path via governance/tests/conftest.py.
    return importlib.import_module('check_test_fallbacks')


def test_find_try_statements_flags_try() -> None:
    mod = _load()
    src = 'def test_x() -> None:\n    try:\n        do()\n    except Exception:\n        pass\n'
    assert mod.find_try_statements(src) == [2]


def test_find_try_statements_passes_pytest_raises() -> None:
    mod = _load()
    src = (
        'import pytest\n'
        'def test_x() -> None:\n'
        '    with pytest.raises(ValueError):\n'
        '        do()\n'
    )
    assert mod.find_try_statements(src) == []
