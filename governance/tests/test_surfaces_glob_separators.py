"""Prove a Surfaces glob means what the issue author reads it to mean.

`fnmatch` does not treat `/` as a separator, so `governance/*` covered the
whole subtree beneath it while rule 7 still reported PASS. The scope contract
is the only thing standing between a slice and an unrelated file, so the
interesting cases here are the ones that must NOT match.
"""
from __future__ import annotations

import fnmatch

import pytest
from slice_gate import path_denied, path_matches


@pytest.mark.parametrize(
    ('path', 'glob'),
    [
        ('governance/slice_gate.py', 'governance/*'),
        ('governance/slice_gate.py', 'governance/*.py'),
        ('governance/slice_gate.py', 'governance/slice_gate.py'),
        ('CHANGELOG.md', '*.md'),
        ('CHANGELOG.md', '*'),
        ('governance/tests/test_x.py', 'governance/**'),
        ('governance/tests/test_x.py', 'governance/**/*.py'),
        # `**/` matches zero directories too.
        ('x.py', '**/x.py'),
        ('a/b/x.py', '**/x.py'),
        ('governance/slice_gate.py', 'governance/slice_?ate.py'),
    ],
)
def test_paths_their_glob_covers(path: str, glob: str) -> None:
    assert path_matches(path, glob)


@pytest.mark.parametrize(
    ('path', 'glob'),
    [
        # The defect: a single star must not cross a separator.
        ('governance/tests/deep.py', 'governance/*'),
        ('governance/tests/deep.py', 'governance/*.py'),
        ('docs/Developer/Configuration.md', 'docs/*'),
        ('governance/slice_gate.py', '*'),
        ('governance/slice_gate.py', '*.py'),
        ('a/b/c.md', '*.md'),
        # `?` is one character, and never the separator.
        ('governance/x/ate.py', 'governance/slice_?ate.py'),
        ('a/b.py', 'a?b.py'),
        # A different subtree entirely.
        ('tests/package/test_x.py', 'governance/**'),
        ('governance.yml', 'governance/**'),
    ],
)
def test_paths_their_glob_must_not_cover(path: str, glob: str) -> None:
    assert not path_matches(path, glob)


def test_a_glob_is_anchored_at_both_ends() -> None:
    """A partial match must not count, or `governance/x.py` would be
    covered by `governance/x` and by `overnance/x.py` alike."""
    assert not path_matches('governance/slice_gate.py', 'governance/slice_gate')
    assert not path_matches('governance/slice_gate.py', 'slice_gate.py')
    assert not path_matches('agovernance/slice_gate.py', 'governance/*.py')


def test_regex_metacharacters_in_a_path_are_literal() -> None:
    """A dot is a dot. Under a naive translation `a.py` would match `axpy`."""
    assert path_matches('a.py', 'a.py')
    assert not path_matches('axpy', 'a.py')
    assert not path_matches('docs/a+b.md', 'docs/a+.md')
    assert path_matches('docs/a+b.md', 'docs/a+b.md')


@pytest.mark.parametrize(
    ('path', 'glob'),
    [
        # The allow-list narrowing is safe; the same narrowing on the deny-list
        # is not, so a deny entry covers everything beneath what it names.
        ('governance/tests/deep.py', 'governance/*'),
        ('governance/tests/deep.py', 'governance/**'),
        ('governance/slice_gate.py', 'governance/*'),
        ('docs/Developer/Configuration.md', 'docs/*'),
        ('docs/Developer/Configuration.md', 'docs'),
        ('governance/tests/fixtures/x.py', 'governance/tests'),
        ('governance/tests/fixtures/x.py', 'governance/tests/'),
    ],
)
def test_out_of_scope_covers_the_subtree_beneath_what_it_names(
    path: str, glob: str
) -> None:
    assert path_denied(path, glob)


@pytest.mark.parametrize(
    ('path', 'glob'),
    [
        # Widening the deny-list must not reach a sibling or another subtree.
        ('governance.yml', 'governance/*'),
        ('governance_extra/x.py', 'governance/*'),
        ('tests/package/test_x.py', 'governance'),
        ('docs.md', 'docs'),
    ],
)
def test_out_of_scope_does_not_reach_beyond_its_own_subtree(
    path: str, glob: str
) -> None:
    assert not path_denied(path, glob)


@pytest.mark.parametrize(
    ('path', 'glob'),
    [
        # Suffix-shaped deny entries: a trailing `*.py` names no directory, so
        # appending `/**` cannot rescue it -- the star itself has to reach.
        ('governance/tests/deep.py', 'governance/*.py'),
        ('docs/a/b/c.md', 'docs/*.md'),
        ('a/b/c.py', '*.py'),
        ('governance/tests/fixtures/x.py', 'governance/*.py'),
    ],
)
def test_a_suffix_shaped_deny_glob_still_reaches_nested_paths(
    path: str, glob: str
) -> None:
    assert path_denied(path, glob)


@pytest.mark.parametrize(
    ('path', 'glob', 'covered'),
    [
        # `fnmatch` honours character classes. Escaping the bracket would make
        # the whole entry a literal that matches no real path -- and on the
        # deny-list, an entry that matches nothing excludes nothing.
        ('governance/ab.py', 'governance/[ab]*.py', True),
        ('a.py', '[ab].py', True),
        ('b.py', '[ab].py', True),
        ('c.py', '[ab].py', False),
        # Negation, spelled `!` by fnmatch and `^` by re.
        ('c.py', '[!ab].py', True),
        ('a.py', '[!ab].py', False),
        # Ranges.
        ('a-b.py', '[a-c]-b.py', True),
        ('z-b.py', '[a-c]-b.py', False),
        # A class does not cross a separator on the allow side.
        ('governance/tests/deep.py', 'governance/[ab]*.py', False),
        # An unterminated bracket is a literal, as in fnmatch.
        ('lit[.py', 'lit[.py', True),
    ],
)
def test_character_classes_behave_as_fnmatch_did(
    path: str, glob: str, covered: bool
) -> None:
    assert path_matches(path, glob) is covered
    assert fnmatch.fnmatch(path, glob) is covered


def test_the_deny_list_is_never_weaker_than_fnmatch() -> None:
    """Rule 8 may block more than `fnmatch` did, never less.

    `fnmatch` is the behaviour this gate shipped with, so anything it excluded
    must stay excluded; a deny-list that quietly stopped blocking a path is the
    one direction of this change that is unsafe.
    """
    cases = [
        ('governance/tests/deep.py', 'governance/*'),
        ('governance/tests/deep.py', 'governance/*.py'),
        ('governance/slice_gate.py', 'governance/*.py'),
        ('docs/a/b/c.md', 'docs/*.md'),
        ('a/b/c.py', '*.py'),
        ('governance/tests/deep.py', 'governance/**'),
        ('x/y.md', '*.md'),
        ('governance/a/b/c/d.py', 'governance/*'),
    ]
    weaker = [
        (path, glob) for path, glob in cases
        if fnmatch.fnmatch(path, glob) and not path_denied(path, glob)
    ]
    assert not weaker, f'deny-list stopped excluding paths fnmatch excluded: {weaker}'


def test_the_deny_list_is_never_weaker_than_the_allow_list() -> None:
    """Anything rule 7 would allow-match, rule 8 must also deny-match.

    Rule 8 is the finer-grained block, so a path the same glob covers on the
    allow side must never slip through on the deny side.
    """
    cases = [
        ('governance/slice_gate.py', 'governance/*'),
        ('governance/tests/deep.py', 'governance/**'),
        ('CHANGELOG.md', '*.md'),
        ('x.py', '**/x.py'),
    ]
    for path, glob in cases:
        assert path_matches(path, glob)
        assert path_denied(path, glob)


def test_the_double_star_that_still_covers_everything() -> None:
    """`**` is genuinely universal, which is why the vacuous-glob guard
    in rule 7 still rejects it as a scope declaration."""
    assert path_matches('anything/at/all.py', '**')
    assert path_matches('top.py', '**')
