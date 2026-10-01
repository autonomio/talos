"""Existing-repository bootstrap never renames Talos or creates a merging PR."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_WORKFLOW = REPO_ROOT / '.github/workflows/bootstrap_repository.yml'


def test_bootstrap_is_manual_and_protected_branch_only() -> None:
    text = BOOTSTRAP_WORKFLOW.read_text()
    data = yaml.safe_load(text)
    assert set(data.get('on', data.get(True))) == {'workflow_dispatch'}
    assert "github.ref == 'refs/heads/master'" in text
    assert 'scripts/configure_repository.py' in text
    assert '--apply' in text
    assert '--files-only' not in text
    assert 'gh pr create' not in text
    assert 'gh pr merge' not in text
    assert 'git push' not in text


def test_bootstrap_stays_bounded_and_serialized() -> None:
    text = BOOTSTRAP_WORKFLOW.read_text()
    assert 'timeout-minutes: 30' in text
    assert 'cancel-in-progress: false' in text
    assert 'REPO_BOOTSTRAP_TOKEN is required' in text


def test_template_rename_bypasses_are_absent() -> None:
    for name in ('ruleset', 'typing', 'version', 'fail_loud'):
        text = (REPO_ROOT / '.github/workflows' / f'pr_checks_{name}.yml').read_text()
        assert 'base != head' not in text
        assert 'new-repository-template' not in text


def test_ruleset_lookup_ignores_organization_rulesets() -> None:
    text = (REPO_ROOT / 'governance/bootstrap_repository.py').read_text()
    assert "item.get('source_type') == 'Repository'" in text
