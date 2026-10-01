"""Code-owner review law: the enforcement surfaces are owned and the
ruleset demands owner approval for changes to them.

An Actions check can be neutered by the PR it judges -- the gate
executes from the judged merge ref. Owner review is the one layer
outside that ref, so the files whose content decides merge verdicts
carry code owners, the ruleset requires their approval on the most
recent push (a stale owner approval must not survive a later push to
an enforcement surface), and there is deliberately no global ``*``
rule (which would make its owner a required approver on every PR
instead of only the enforcement surfaces). Every owned path must
resolve on disk: a renamed or mistyped surface would otherwise sit
unowned while GitHub silently matches nothing.
"""
from __future__ import annotations

import json
import re

from _common import REPO_ROOT

CODEOWNERS = REPO_ROOT / '.github' / 'CODEOWNERS'
RULESET_SNAPSHOT = REPO_ROOT / '.github' / 'rulesets' / 'master.json'
ENFORCEMENT_PATHS = frozenset({
    '/CLAUDE.md',
    '/AGENTS.md',
    '/AUTONOMIO_PR_GUIDELINE.md',
    '/TALOS_REPO_SPECIFICS.md',
    '/scripts/',
    '/governance/',
    '/.github/',
    '/governance.yml',
    '/pyproject.toml',
    '/requirements/',
})
EXPECTED_OWNERS = frozenset({'@mikkokotila'})


def _rule_lines() -> list[list[str]]:
    lines = []
    for raw in CODEOWNERS.read_text(encoding='utf-8').splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith('#'):
            lines.append(stripped.split())
    return lines


def test_codeowners_covers_enforcement_surfaces() -> None:
    rules = _rule_lines()
    assert rules, f'{CODEOWNERS} declares no ownership rules'
    covered = {rule[0] for rule in rules}
    assert covered == ENFORCEMENT_PATHS
    for rule in rules:
        assert set(rule[1:]) == EXPECTED_OWNERS, rule[0]
        assert all(re.fullmatch(r'@[\w-]+', owner) for owner in rule[1:]), rule


def test_codeowners_paths_resolve_on_disk() -> None:
    for rule in _rule_lines():
        rel = rule[0].lstrip('/')
        target = REPO_ROOT / rel.rstrip('/')
        if rule[0].endswith('/'):
            assert target.is_dir(), f'{rule[0]} does not resolve to a directory'
        else:
            assert target.is_file(), f'{rule[0]} does not resolve to a file'


def test_codeowners_has_no_global_rule() -> None:
    assert all(rule[0] != '*' for rule in _rule_lines()), (
        'a global * rule makes its owner a required approver on every PR'
    )


def test_ruleset_snapshot_requires_code_owner_review() -> None:
    snapshot = json.loads(RULESET_SNAPSHOT.read_text(encoding='utf-8'))
    params = [
        rule['parameters']
        for rule in snapshot['rules']
        if rule['type'] == 'pull_request'
    ]
    assert [p['require_code_owner_review'] for p in params] == [True]
    # A code-owner approval granted on one revision must not survive a
    # later push: the most recent reviewable push needs approval from
    # someone other than its pusher.
    assert [p['require_last_push_approval'] for p in params] == [True]
