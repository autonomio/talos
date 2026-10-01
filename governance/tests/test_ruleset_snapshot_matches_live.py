"""The committed ruleset snapshot records protections, never relaxations.

Law 10 makes `.github/rulesets/master.json` the declaration that live branch
protection must match. When the two disagree the question is always which one
is wrong, and syncing the snapshot to the live state is the move that can
launder a weakening into law: if someone loosened branch protection out of
band, copying the live payload over the snapshot makes the loosening the new
written rule and `pr_checks_ruleset` goes quiet.

So the sync direction carries assertions rather than a judgement call. Every
protection flag this repository relies on is pinned on here, and `bypass_actors`
is pinned empty. A future snapshot sync that drags any of them the wrong way
fails in `pr_checks_lint` rather than passing as routine housekeeping.

`bypass_actors` matters most and is the one thing `pr_checks_ruleset` cannot
see at PR time -- only the post-merge `audit_main_ruleset` reads it live. This
pins the snapshot half of that pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO_ROOT / '.github/rulesets/master.json'
CODEQL_PRESENT = 'PR Checks CodeQL (python)' in (REPO_ROOT / 'CLAUDE.md').read_text(encoding='utf-8')

# Every `pull_request` flag whose value is a protection. Each must stay true.
PROTECTION_FLAGS = (
    'require_code_owner_review',
    'require_extra_approval_for_unattributed_changes',
    'require_last_push_approval',
    'required_review_thread_resolution',
)


def _pull_request_parameters() -> dict[str, object]:
    payload = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    rule = next(r for r in payload['rules'] if r['type'] == 'pull_request')
    params = rule['parameters']
    assert isinstance(params, dict)
    return params


def test_snapshot_requires_extra_approval_for_unattributed_changes() -> None:
    """The field GitHub added to the ruleset API, recorded as it runs live.

    Its absence from the snapshot is what made every open PR fail
    `pr_checks_ruleset` with `ruleset drift detected`.
    """
    assert _pull_request_parameters()['require_extra_approval_for_unattributed_changes'] is True


def test_every_protection_flag_stays_on() -> None:
    """Syncing a snapshot must never turn a protection off.

    This is the assertion that makes the sync safe rather than merely quiet:
    a future sync that picks up a loosened live ruleset fails here.
    """
    params = _pull_request_parameters()
    off = [flag for flag in PROTECTION_FLAGS if params.get(flag) is not True]
    assert not off, f'protection flags no longer required by the snapshot: {off}'
    assert params['required_approving_review_count'] >= 1


def test_snapshot_grants_no_bypass_actors() -> None:
    """Nobody may skip the rules.

    `bypass_actors` is invisible to the PR-time ruleset gate; only the
    post-merge audit reads it live. Pinning the snapshot side means a bypass
    entry cannot arrive as part of a routine snapshot sync.
    """
    payload = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    assert payload.get('bypass_actors', []) == []


def test_snapshot_still_protects_main_actively() -> None:
    """A snapshot that targets nothing, or sits disabled, protects nothing."""
    payload = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    assert payload['enforcement'] == 'active'
    assert 'refs/heads/master' in payload['conditions']['ref_name']['include']


@pytest.mark.skipif(not CODEQL_PRESENT, reason='CodeQL explicitly removed from this repository')
def test_snapshot_blocks_every_new_codeql_security_alert() -> None:
    """Analysis success alone cannot enforce the security finding contract."""
    payload = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    rules = [rule for rule in payload['rules'] if rule['type'] == 'code_scanning']
    assert len(rules) == 1, 'a native CodeQL findings rule must accompany analysis'
    assert rules[0]['parameters']['code_scanning_tools'] == [{
        'tool': 'CodeQL',
        'security_alerts_threshold': 'all',
        'alerts_threshold': 'none',
    }]
