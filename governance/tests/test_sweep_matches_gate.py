"""The scheduled sweep must reach the same verdict as the gate it re-runs.

The sweep exists because `pr_checks_slice` computes its verdict from the
judged PR's own merge ref; re-running the gate from `main` is what closes that
self-attestation channel. Publishing a *different* answer than the gate is
therefore not a second opinion -- it is the sweep being wrong, because it runs
the same gate over the same PR.

It was wrong. The sweep gathered the PR's title, body and files but not its
author, so `exit_if_bot_exempt` received an empty string, enforced, and posted
a failure over the `SKIP` the pull_request run had correctly reported an hour
earlier. Eleven Dependabot PRs sat blocked by a required check that had passed
and then turned red on a timer.

These tests pin argument parity rather than the presence of one flag: the next
argument added to one call site and forgotten in the other fails here, instead
of an hour after the PR goes green.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP = REPO_ROOT / '.github/workflows/pr_checks_slice_sweep.yml'
GATE_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_slice.yml'

# Flags that carry a judgement input. `--pr-author` decides the automation
# exemption; the rest decide the scope and closing-set checks. Both call sites
# must pass every one of them or they can disagree.
_FLAG_RE = re.compile(r'(--[a-z-]+)')


def _gate_invocation(workflow: Path) -> set[str]:
    """The flags one workflow passes to `slice_gate.py`.

    Anchored on the invocation rather than the first mention of the filename:
    both workflows name the module in a comment before they run it, and
    starting there yields an empty flag set that makes the parity assertion
    vacuously compare nothing.
    """
    text = workflow.read_text(encoding='utf-8')
    start = text.index('python governance/slice_gate.py')
    # The invocation ends at the first line that is not a continuation.
    tail = text[start:]
    lines: list[str] = []
    for line in tail.splitlines()[1:]:
        lines.append(line)
        if not line.rstrip().endswith('\\'):
            break
    return set(_FLAG_RE.findall('\n'.join(lines)))


def test_sweep_reads_the_author_from_the_rest_user_login() -> None:
    """The author must be read the way the pull_request event carries it.

    `gh pr view --json author` serialises a GitHub App as `app/dependabot`;
    the pull_request event, and therefore `automation.bot_authors`, carries
    `dependabot[bot]`. Passing the first form is worse than passing nothing:
    the gate receives a non-empty author that matches no configured bot, so
    it enforces and overturns a correct SKIP while looking correctly wired.
    """
    text = SWEEP.read_text(encoding='utf-8')
    assert ".user.login" in text, (
        'the sweep must read the author from the REST pull request payload, '
        'which carries the same login the pull_request event does'
    )
    assert '--json title,body,changedFiles,author' not in text, (
        "`gh pr view --json author` yields `app/<slug>` for a GitHub App, "
        "which never matches the `<slug>[bot]` form in automation.bot_authors"
    )


def test_configured_bot_authors_are_in_event_login_form() -> None:
    """The configured authors must be the form the gate actually receives.

    Pins the other half of the same mismatch: a `bot_authors` entry written as
    `app/dependabot` would match the sweep's old source and never match the
    pull_request run, splitting the two verdicts the other way.
    """
    import yaml

    config = yaml.safe_load((REPO_ROOT / 'governance.yml').read_text(encoding='utf-8'))
    for author in config.get('automation', {}).get('bot_authors', []):
        assert not author.startswith('app/'), (
            f'automation.bot_authors lists {author!r}; the pull_request event '
            f'carries the `<slug>[bot]` form, so an `app/` entry never matches'
        )


def test_sweep_passes_the_pr_author() -> None:
    """The gathered author must actually reach the gate."""
    assert '--pr-author' in _gate_invocation(SWEEP), (
        'the sweep runs the slice gate without --pr-author, so the gate sees an '
        'empty author, enforces, and publishes a failure over a correct SKIP'
    )


def test_sweep_and_gate_pass_the_same_arguments() -> None:
    """Argument parity, not the presence of one flag.

    Two call sites running the same gate over the same PR must hand it the
    same inputs. Asserting only `--pr-author` would pass while some later
    argument is added to one and forgotten in the other.
    """
    sweep = _gate_invocation(SWEEP)
    gate = _gate_invocation(GATE_WORKFLOW)
    assert sweep == gate, (
        f'the sweep and the pull_request gate disagree on inputs; '
        f'sweep-only: {sorted(sweep - gate)}, gate-only: {sorted(gate - sweep)}. '
        f'Differing inputs mean the two can reach different verdicts on the '
        f'same pull request.'
    )


def test_an_empty_author_enforces_rather_than_skips() -> None:
    """An unknown author is judged, not waved through.

    The empty-author default is the safe direction and must stay that way:
    a sweep that could not determine an author should enforce every law.
    """
    import importlib

    common = importlib.import_module('_common')
    # Returns without raising SystemExit -> the gate carries on and enforces.
    assert common.exit_if_bot_exempt('slice', '', 'TEST') is None
