from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CLOSEOUT_GUARD_WORKFLOW = REPO_ROOT / '.github/workflows/slice_closeout_guard.yml'
ON_ISSUE_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_slice_on_issue.yml'
SWEEP_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_slice_sweep.yml'
READINESS_WORKFLOW = REPO_ROOT / '.github/workflows/pr_merge_readiness.yml'
RULESET_SNAPSHOT = REPO_ROOT / '.github/rulesets/master.json'


def test_post_merge_changelog_workflow_removed() -> None:
    assert not (REPO_ROOT / '.github/workflows/pr_post_changelog.yml').exists()


def test_slice_closeout_guard_workflow_contract() -> None:
    workflow = CLOSEOUT_GUARD_WORKFLOW.read_text(encoding='utf-8')

    assert 'issues:\n    types: [closed]' in workflow
    # Parsed rather than string-matched: the condition is a multi-line
    # folded scalar since it also excludes withdrawals, so pinning its exact
    # source text breaks on a reformat that changes nothing semantic.
    condition = yaml.safe_load(workflow)['jobs']['slice_closeout_guard']['if']
    assert "contains(github.event.issue.labels.*.name, 'slice')" in condition
    assert 'issues: write' in workflow
    assert 'closedByPullRequestsReferences' in workflow
    # Evidence quality: only successful check runs may become closeout
    # evidence — a failed required check must fail the writer loud.
    assert "select(.conclusion == \"success\")" in workflow
    # An appended duplicate Done Means section must not receive (or
    # shadow) evidence, and a hand-typed Merge SHA on a no-PR close
    # must be reachable from main to count.
    # The splice moved into governance/closeout_evidence.py so it could be
    # unit tested; the workflow now calls it. Its fail-closed behaviour is
    # asserted there, in test_closeout_evidence.py.
    assert 'python governance/closeout_evidence.py' in workflow
    evidence = (REPO_ROOT / 'governance/closeout_evidence.py').read_text(encoding='utf-8')
    assert 'expected exactly one Done Means section' in evidence
    assert 'compare/master...$CLAIMED' in workflow
    assert r"r'^##+ Done Means\b.*?^##+ Author Checks\b'" in workflow
    # Fill: a merged closing PR gets the evidence fields written in place.
    assert "if: steps.closing_pr.outputs.pr_number != ''" in workflow
    assert 'gh issue edit "$ISSUE_NUMBER" --repo "$GITHUB_REPOSITORY" --body-file new_body.txt' in workflow
    # Verify: an incomplete-evidence close with no merged PR, or any
    # guard failure, reopens the issue instead of letting the close
    # stand.
    assert 'gh issue reopen' in workflow
    assert 'empty Merged PR number' in workflow
    assert 'empty required-run list' in workflow
    assert 'if: failure()\n' in workflow


def test_slice_on_issue_workflow_contract() -> None:
    workflow = ON_ISSUE_WORKFLOW.read_text(encoding='utf-8')

    # Rerun-first delivery: heal the canonical pull_request run instead
    # of stacking parallel check-runs; the API POST stays only as the
    # fail-closed fallback.
    assert 'actions/runs/$RUN_ID/rerun' in workflow
    assert 'falling back to check-run POST' in workflow
    assert 'actions: write' in workflow
    assert 'check-runs?check_name=pr_checks_slice' in workflow
    # API-posted reconciliation: an API-posted check-run outranks the
    # workflow-run entries of the same name and only a newer POST
    # supersedes it (Autonomio/Limen PR #757), so a disagreeing latest
    # API-posted check-run always draws a superseding POST.
    assert 'check_suite_id' in workflow
    assert 'LATEST_POSTED' in workflow
    # Rule 9 staleness: the affected set includes the parent PRD's
    # slice sub-issue siblings, and a parent-lookup failure fails loud
    # instead of silently shrinking the set.
    assert 'sub_issues' in workflow
    assert 'refusing to compute a sibling set that may be incomplete' in workflow
    # SIGPIPE-safe truncation: parameter expansion, never a pipe.
    assert 'SUMMARY=${SUMMARY:0:60000}' in workflow
    assert '"$GATE_OUT" | head -c' not in workflow


def test_slice_sweep_workflow_contract() -> None:
    workflow = SWEEP_WORKFLOW.read_text(encoding='utf-8')

    assert "- cron: '17 */6 * * *'" in workflow
    assert 'workflow_dispatch:' in workflow
    # A dispatch from another ref must still execute main's gate.
    assert 'ref: master' in workflow
    # Killing a sweep mid-posting would strand some PRs a full interval.
    assert 'cancel-in-progress: false' in workflow
    # Main-only authority: the sweep publishes its own verdict against
    # the latest check-run and never reruns the pull_request workflow —
    # a rerun would re-execute the gate from the judged PR's merge ref
    # and let the PR self-attest.
    assert 'check-runs?check_name=pr_checks_slice' in workflow
    assert 'self-attest' in workflow
    assert 'actions/runs/$RUN_ID/rerun' not in workflow
    assert 'actions: write' not in workflow
    # API-posted reconciliation (Autonomio/Limen PR #757): a disagreeing
    # latest API-posted check-run always draws a superseding POST, and
    # classifying check-runs by canonical check suite needs actions:
    # read on the token.
    assert 'check_suite_id' in workflow
    assert 'LATEST_POSTED' in workflow
    assert 'actions: read' in workflow
    assert 'file enumeration incomplete' in workflow
    assert 'SUMMARY=${SUMMARY:0:60000}' in workflow
    assert '--require-hashes -r requirements/ci/gate-tools.txt' in workflow


def test_merge_readiness_workflow_contract() -> None:
    workflow = READINESS_WORKFLOW.read_text(encoding='utf-8')

    # pull_request_review_thread is a webhook event only, not an
    # Actions trigger — the defect that stalled the downstream first
    # delivery; comment activity and suite completions refresh instead.
    # Matched as a trigger key so the workflow's own explanatory
    # comment does not satisfy the assertion.
    assert '\n  pull_request_review_thread:' not in workflow
    assert 'pull_request_review:' in workflow
    assert 'pull_request_review_comment:' in workflow
    assert 'check_suite:' in workflow
    assert '<!-- merge-readiness -->' in workflow

    # Queued, not cancelled. The group exists to serialise the
    # read-then-create on the marker comment, and cancelling is not
    # serialising -- it kills the earlier run, which then surfaces as a
    # failed check on the PR this workflow exists to report on.
    parsed = yaml.safe_load(workflow)
    assert parsed['concurrency']['cancel-in-progress'] is False
    assert 'required-check inventory unavailable (fail-closed)' in workflow
    assert 'pull-requests: write' in workflow
    # One concurrency lane per PR across every event type, so parallel
    # runs cannot race the read-then-create on the marker comment.
    assert 'github.event.check_suite.pull_requests[0].number' in workflow
    # Informational only: never itself a required context, so it cannot
    # deadlock the merge it reports on.
    snapshot = json.loads(RULESET_SNAPSHOT.read_text(encoding='utf-8'))
    for rule in snapshot['rules']:
        if rule['type'] == 'required_status_checks':
            contexts = [c['context'] for c in rule['parameters']['required_status_checks']]
            assert 'pr_merge_readiness' not in contexts


def test_update_changelog_script_removed() -> None:
    assert not (REPO_ROOT / 'governance/update_changelog.py').exists()


def test_typing_gate_setup_failures_exit_2() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / 'repo'
        shutil.copytree(
            REPO_ROOT,
            tmp,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('.git'),
        )
        (tmp / 'pyproject.toml').write_text('not = [valid\n', encoding='utf-8')

        result = subprocess.run(
            [
                # sys.executable, not a bare python3: below 3.11 the gate
                # resolves tomli, and only the interpreter running this
                # suite has it installed.
                sys.executable,
                'governance/typing_gate.py',
                '--pyright-json',
                '/tmp/missing-pyright.json',
                '--bootstrap',
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=tmp,
        )

    assert result.returncode == 2
    assert result.stdout == ''
    assert 'typing_gate: cannot parse pyproject.toml:' in result.stderr


def test_closeout_guard_skips_withdrawals() -> None:
    """A `not planned` close is a withdrawal, and the guard must let it stand.

    The guard reopens any close it cannot back with a merge SHA, a merged PR
    number and a required-run list. A withdrawn slice has none of those and
    never will, so guarding it reopens the issue forever -- which is what
    happened to #70-#74. `state_reason` is the durable record of the decision.
    """
    workflow = yaml.safe_load(CLOSEOUT_GUARD_WORKFLOW.read_text(encoding='utf-8'))
    condition = workflow['jobs']['slice_closeout_guard']['if']
    assert 'state_reason' in condition
    assert "!= 'not_planned'" in condition
    # a duplicate close is equally evidence-free and equally permanent
    assert "!= 'duplicate'" in condition
    # every other close still goes through the evidence check
    assert "contains(github.event.issue.labels.*.name, 'slice')" in condition


def test_version_gate_has_no_permanent_bypass() -> None:
    text = (REPO_ROOT / '.github/workflows/pr_checks_version.yml').read_text()
    assert 'base != head' not in text
    assert 'governance/version_gate.py' in text
    assert '--base-version-source' in text
    assert '--head-version-source' in text


def test_ratchet_gates_have_no_permanent_bootstrap_mode() -> None:
    for name in ('pr_checks_typing', 'pr_checks_fail_loud', 'pr_checks_ruleset'):
        text = (REPO_ROOT / f'.github/workflows/{name}.yml').read_text()
        assert 'base != head' not in text
        assert 'new-repository-template' not in text
        assert '--bootstrap' in text or name == 'pr_checks_ruleset'
