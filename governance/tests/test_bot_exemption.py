"""Prove the bot exemption skips exactly what it claims and nothing else.

The exemption is the one place a gate is allowed to not run, so the interesting
cases are the ones where it must still run: a bot on a gate nobody exempted, a
human whose login was never listed, and a repository that configured nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import _common
import pytest
import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / 'governance.yml'
WORKFLOWS = REPO_ROOT / '.github' / 'workflows'

_EXEMPTING = """
automation:
  bot_authors:
    - dependabot[bot]
  exempt_gates:
    - slice
    - version
"""


@pytest.fixture
def configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point `_common` at a scratch `governance.yml` the test writes."""
    path = tmp_path / 'governance.yml'
    monkeypatch.setattr(_common, 'GOVERNANCE_CONFIG', path)
    yield path


def test_a_listed_bot_skips_a_listed_gate(configured: Path) -> None:
    configured.write_text(_EXEMPTING, encoding='utf-8')
    with pytest.raises(SystemExit) as exc:
        _common.exit_if_bot_exempt('slice', 'dependabot[bot]', 'SLICE GATE')
    assert exc.value.code == 0


def test_a_listed_bot_still_faces_an_unlisted_gate(configured: Path) -> None:
    """`exempt_gates` is the whole allow-list; `typing` is not on it."""
    configured.write_text(_EXEMPTING, encoding='utf-8')
    _common.exit_if_bot_exempt('typing', 'dependabot[bot]', 'TYPING GATE')


def test_an_unlisted_author_faces_every_gate(configured: Path) -> None:
    configured.write_text(_EXEMPTING, encoding='utf-8')
    _common.exit_if_bot_exempt('slice', 'mikkokotila', 'SLICE GATE')


def test_an_absent_author_enforces(configured: Path) -> None:
    """An empty `--pr-author` must enforce, not exempt.

    The workflows fill it from the event payload; a payload shape that stopped
    carrying the login would otherwise silently disable two laws.
    """
    configured.write_text(_EXEMPTING, encoding='utf-8')
    _common.exit_if_bot_exempt('slice', '', 'SLICE GATE')


def test_a_repository_that_configures_nothing_enforces(configured: Path) -> None:
    """Both lists default to empty, so a missing section cannot exempt."""
    configured.write_text('repository:\n  name: x\n', encoding='utf-8')
    _common.exit_if_bot_exempt('slice', 'dependabot[bot]', 'SLICE GATE')


def test_an_emptied_exempt_list_restores_both_laws(configured: Path) -> None:
    """The documented way to hold bots to laws 1 and 5 must actually work."""
    configured.write_text(
        'automation:\n  bot_authors:\n    - dependabot[bot]\n  exempt_gates: []\n',
        encoding='utf-8',
    )
    _common.exit_if_bot_exempt('slice', 'dependabot[bot]', 'SLICE GATE')
    _common.exit_if_bot_exempt('version', 'dependabot[bot]', 'VERSION GATE')


def test_a_misshapen_list_blocks_rather_than_guessing(configured: Path) -> None:
    """A string where a list belongs must fail setup, not be iterated.

    `'slice' in 'slice,version'` is true for a bare string, so a shape that
    looked close enough would exempt gates nobody listed.
    """
    configured.write_text(
        "automation:\n  bot_authors: dependabot[bot]\n  exempt_gates:\n    - slice\n",
        encoding='utf-8',
    )
    with pytest.raises(SystemExit) as exc:
        _common.exit_if_bot_exempt('slice', 'dependabot[bot]', 'SLICE GATE')
    assert exc.value.code != 0


def test_a_configured_automation_section_names_only_real_gates() -> None:
    """Check the shape, never the policy.

    Asserting the shipped values would ship a failing test to every derived
    repository that took the documented advice: `exempt_gates: []` and omitting
    `automation` entirely are both supported, and an equality assertion here
    would turn either into an unmergeable branch. What must hold everywhere is
    that an exemption cannot name a gate that does not exist -- a typo there
    would look configured and exempt nothing.
    """
    data = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    automation = data.get('automation')
    if automation is not None:
        bot_authors = automation.get('bot_authors', [])
        exempt_gates = automation.get('exempt_gates', [])
        assert isinstance(bot_authors, list)
        assert isinstance(exempt_gates, list)
        unknown = sorted(set(exempt_gates) - set(data.get('gates', {})))
        assert not unknown, (
            f'automation.exempt_gates names gates that do not exist: {unknown}. '
            f'A misspelled gate reads as configured and exempts nothing.'
        )


def test_both_exempted_gates_are_told_who_opened_the_pull_request() -> None:
    """An exemption the workflow never passes an author to is inert.

    Without `--pr-author` the gate defaults to enforcing, so a missing flag
    would leave dependabot blocked while the config claimed otherwise.
    """
    for workflow, flag in (
        ('pr_checks_slice.yml', '--pr-author'),
        ('pr_checks_version.yml', '--pr-author'),
    ):
        text = (WORKFLOWS / workflow).read_text(encoding='utf-8')
        assert flag in text, f'{workflow} never passes {flag}'
        assert 'github.event.pull_request.user.login' in text, (
            f'{workflow} passes {flag} but not the login that fills it'
        )


def test_both_laws_disclose_the_exemption() -> None:
    """A law that does not mention its own exemption overstates what runs."""
    laws = (REPO_ROOT / 'CLAUDE.md').read_text(encoding='utf-8')
    for law in ('pr_checks_slice', 'pr_checks_version'):
        line = next(ln for ln in laws.splitlines() if ln.endswith(f'*({law})*'))
        assert 'automation.bot_authors' in line, (
            f'the law ending in ({law}) does not disclose the bot exemption'
        )


def test_the_gates_wire_the_exemption_to_their_own_name() -> None:
    """A gate passing another gate's name would exempt the wrong law."""
    for module, name in (('slice_gate.py', 'slice'), ('version_gate.py', 'version')):
        text = (REPO_ROOT / 'governance' / module).read_text(encoding='utf-8')
        assert f"exit_if_bot_exempt('{name}', args.pr_author" in text, (
            f'{module} does not exempt under its own gate name'
        )


def test_the_helper_is_importable_the_way_the_gates_import_it() -> None:
    """The gates run as scripts with `governance/` on the path."""
    assert str(REPO_ROOT / 'governance') in sys.path
    assert callable(_common.exit_if_bot_exempt)
