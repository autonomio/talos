"""Property tests for the parsers fed arbitrary user-authored text.

The slice-gate extractors are the only functions here handed untrusted input:
a slice issue body is written by whoever opens the issue, and `slice_gate` is
a required merge gate. A parser that raises on a malformed body turns an
authoring mistake into a crashed gate, which reads as infrastructure failure
rather than as the mistake it is.

The invariant is deliberately narrow: these extractors must always return,
never raise, and must return the declared type. *What* they return on nonsense
is the gate's business; *that* they return at all is this file's.

Coverage-guided fuzzing (Atheris) was the first approach and was withdrawn:
it ships no wheels for Python 3.12 or later and cannot build against this
repository's floor, so the workflow could not install its own dependency.
Property-based testing reaches the same invariant, runs inside an already
required check rather than a separate advisory workflow, and shrinks a failing
case to a minimal reproduction — which a 60-second fuzz run does not.
"""
from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slice_gate import (
    extract_out_of_scope_globs,
    extract_surfaces_globs,
    find_closing_references,
)

# Fragments drawn from real issue bodies, so generated text explores the
# structure the parsers actually branch on rather than uniform noise.
FRAGMENTS = st.sampled_from([
    '## Surfaces',
    '## Out of Scope',
    'Modified:',
    'Added:',
    'Removed:',
    '- `path/to/file.py`',
    '- (none)',
    '-',
    '- ``',
    'Closes #12',
    'Fixes #0',
    'Resolves #999999999999',
    '> **Significance.**',
    '`',
    '\x00',
    '\u2028',  # LINE SEPARATOR
    '\r\n',
    '',
])

BODIES = st.one_of(
    st.text(),
    st.lists(FRAGMENTS, max_size=40).map('\n'.join),
)


@given(body=BODIES)
@settings(max_examples=400, deadline=None)
def test_surfaces_extractor_always_returns_a_list_of_strings(body: str) -> None:
    """Verify the Surfaces extractor never raises on arbitrary input."""
    result = extract_surfaces_globs(body)
    assert isinstance(result, list)
    assert all(isinstance(entry, str) for entry in result)


@given(body=BODIES)
@settings(max_examples=400, deadline=None)
def test_out_of_scope_extractor_always_returns_a_list_of_strings(body: str) -> None:
    """Verify the Out of Scope extractor never raises on arbitrary input."""
    result = extract_out_of_scope_globs(body)
    assert isinstance(result, list)
    assert all(isinstance(entry, str) for entry in result)


@given(body=BODIES)
@settings(max_examples=400, deadline=None)
def test_closing_reference_finder_always_returns_issue_numbers(body: str) -> None:
    """Verify the closing-reference finder never raises and yields integers."""
    result = find_closing_references(body)
    assert isinstance(result, list)
    assert all(isinstance(number, int) for number in result)


@given(body=BODIES)
@settings(max_examples=200, deadline=None)
def test_extractors_never_return_the_placeholder_entry(body: str) -> None:
    """Verify `(none)` is filtered rather than treated as a path.

    A body carrying the template's own `- (none)` placeholder must yield no
    glob. Returning it would make the scope check compare real paths against
    a literal that matches nothing, which fails closed on a valid issue.
    """
    for extractor in (extract_surfaces_globs, extract_out_of_scope_globs):
        assert '(none)' not in extractor(body)
        assert 'none' not in extractor(body)
