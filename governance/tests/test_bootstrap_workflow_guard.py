"""Administrative bootstrap requires explicit dispatch and branch scope."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / '.github/workflows/bootstrap_repository.yml'


def test_dispatch_is_the_only_trigger() -> None:
    payload = yaml.safe_load(WORKFLOW.read_text())
    assert set(payload.get('on', payload.get(True))) == {'workflow_dispatch'}


def test_job_guard_restricts_the_protected_branch() -> None:
    payload = yaml.safe_load(WORKFLOW.read_text())
    assert payload['jobs']['bootstrap']['if'] == "github.ref == 'refs/heads/master'"


def test_bootstrap_does_not_rewrite_or_merge_repository_files() -> None:
    text = WORKFLOW.read_text()
    assert 'scripts/configure_repository.py' in text
    assert '--apply' in text
    assert '--files-only' not in text
    assert 'gh pr merge' not in text
    assert 'git push' not in text
