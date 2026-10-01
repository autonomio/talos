"""Keep absent frameworks distinct from broken installed backend dependencies."""
from types import ModuleType

import pytest

import talos
from talos._optional import require_optional


def test_optional_import_returns_installed_module():
    """Resolve an available dependency without loading a training backend."""
    module = require_optional('math', 'Math', 'pytorch')
    assert isinstance(module, ModuleType)
    assert module.sqrt(9) == 3


def test_absent_framework_identifies_extra_and_preserves_cause():
    """Tell core-only users which extra supplies the missing framework."""
    with pytest.raises(ImportError, match=r'pip install talos\[pytorch\]') as caught:
        require_optional('talos_absent_backend.layers', 'PyTorch', 'pytorch')
    cause = caught.value.__cause__
    assert isinstance(cause, ModuleNotFoundError)
    assert cause.name == 'talos_absent_backend'


@pytest.mark.parametrize(
    ('source', 'error_type', 'message'),
    [
        ('import absent_backend_dependency\n', ModuleNotFoundError, 'absent_backend_dependency'),
        ('import json.absent_component\n', ModuleNotFoundError, 'json.absent_component'),
        ('raise ImportError("native backend ABI mismatch")\n', ImportError, 'ABI mismatch'),
    ],
)
def test_installed_backend_failure_is_not_rewritten(
    tmp_path, monkeypatch, source, error_type, message,
):
    """Expose transitive and native import faults with their original diagnosis."""
    (tmp_path / 'installed_backend.py').write_text(source, encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(error_type, match=message) as caught:
        require_optional('installed_backend', 'PyTorch', 'pytorch')
    assert 'pip install' not in str(caught.value)
    assert caught.value.__cause__ is None


def test_unknown_public_attribute_fails_without_importing_backend():
    """Reject unknown Python API names through the public lazy interface."""
    with pytest.raises(AttributeError, match='missing_talos_surface'):
        getattr(talos, 'missing_talos_surface')
