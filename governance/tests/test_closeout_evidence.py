"""Execute the closeout evidence writer against real issue-body shapes.

The splice used to live inline in `slice_closeout_guard.yml`, where it could
not be unit tested -- so a placement bug (evidence appended after the Author
Checks header rather than before it) was only findable by hand. These tests
exercise the extracted function directly, which is the point of extracting it.
"""
from __future__ import annotations

from typing import Final

import pytest
from closeout_evidence import insert_evidence

SHA: Final[str] = 'a' * 40
PR: Final[str] = '999'
RUNS: Final[list[str]] = ['- pr_checks_lint : 111', '- pr_checks_tests : 222']

SCAFFOLDED: Final[str] = """## Done Means

- [x] Capability complete

Merge SHA:
Merged PR number:
Required CI runs (workflow name : run id):
-
-

## Author Checks

- [ ] a box
"""

UNSCAFFOLDED: Final[str] = """## Done Means

- [x] Capability complete

## Author Checks

- [ ] a box
"""

PARTIAL: Final[str] = """## Done Means

- [x] Capability complete

Merge SHA:

## Author Checks

- [ ] a box
"""


def _done_means(body: str) -> str:
    return body.split('## Author Checks')[0]


@pytest.mark.parametrize(
    ('name', 'body'),
    [('scaffolded', SCAFFOLDED), ('unscaffolded', UNSCAFFOLDED), ('partial', PARTIAL)],
)
def test_evidence_lands_inside_done_means(name: str, body: str) -> None:
    """Every field ends up in Done Means, whatever scaffolding the body had.

    The unscaffolded and partial shapes are the regression: appending to the
    end of the matched section puts the evidence under Author Checks, because
    the section match includes the header that terminates it.
    """
    result = insert_evidence(body, SHA, PR, RUNS)
    section = _done_means(result)
    assert f'Merge SHA: {SHA}' in section, name
    assert f'Merged PR number: #{PR}' in section, name
    assert 'pr_checks_lint : 111' in section, name
    assert 'pr_checks_tests : 222' in section, name


@pytest.mark.parametrize(
    ('name', 'body'),
    [('scaffolded', SCAFFOLDED), ('unscaffolded', UNSCAFFOLDED), ('partial', PARTIAL)],
)
def test_nothing_lands_after_author_checks(name: str, body: str) -> None:
    """The Author Checks section keeps its own content and gains none."""
    tail = insert_evidence(body, SHA, PR, RUNS).split('## Author Checks', 1)[1]
    assert 'Merge SHA' not in tail, name
    assert 'Merged PR number' not in tail, name
    assert 'Required CI runs' not in tail, name
    assert '- [ ] a box' in tail, name


def test_fields_are_written_once() -> None:
    """A scaffolded body has its placeholders filled, not duplicated."""
    result = insert_evidence(SCAFFOLDED, SHA, PR, RUNS)
    assert result.count('Merge SHA:') == 1
    assert result.count('Merged PR number:') == 1
    assert result.count('Required CI runs') == 1


def test_stale_run_lines_are_replaced() -> None:
    """A rerun overwrites the previous run ids rather than appending to them."""
    once = insert_evidence(SCAFFOLDED, SHA, PR, RUNS)
    twice = insert_evidence(once, SHA, PR, ['- pr_checks_lint : 333'])
    assert '333' in twice
    assert '111' not in twice
    assert '222' not in twice


def test_ambiguous_section_fails_closed() -> None:
    """Two Done Means sections are contradicted structure, not absent fields."""
    with pytest.raises(ValueError, match='exactly one Done Means section'):
        insert_evidence(SCAFFOLDED + SCAFFOLDED, SHA, PR, RUNS)


def test_absent_section_fails_closed() -> None:
    """A body with no Done Means section is not silently left unchanged."""
    with pytest.raises(ValueError, match='exactly one Done Means section'):
        insert_evidence('## Something Else\n\ntext\n', SHA, PR, RUNS)
