"""Pin the universal PR guideline; hold the repo appendix to its shape.

The universal guideline is identical in every repository, so a byte pin is the
right instrument: it is the same artifact everywhere and any edit should be a
deliberate, visible act.

The repo-specific appendix is not. Bootstrap rewrites it -- that is what makes
it repo-specific -- so byte-pinning it meant the digest test failed in every
repository created from this template, inside a required check, before the
bootstrap PR could go green. A file named `REPO_SPECIFICS` that cannot differ
per repository was never going to hold.

So the appendix is checked for the properties that must survive rewriting: it
exists, it points at the guideline, and it still carries repo-scoped entries.
What those entries say is each repository's own business.
"""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PR_GUIDELINE = REPO_ROOT / 'AUTONOMIO_PR_GUIDELINE.md'
REPO_SPECIFICS = REPO_ROOT / 'TALOS_REPO_SPECIFICS.md'
EXPECTED_PR_GUIDELINE_SHA256 = 'a73a244de316619e42634de6076f44b9b91b42af0505523fe8647a73b7de61db'

# Tokens the bootstrap rename engine rewrites. A file carrying any of them
# cannot also carry a byte pin.
#
# Assembled rather than written as literals: this file is itself inside the
# rewrite sweep, so a literal `talos` here becomes the
# derived repository's own package name and the check silently stops looking
# for the thing it was written to find. The seed names come from the bootstrap
# module, which is the one file the sweep skips.
_SEEDS = frozenset({'new' + '_repository_' + 'template', 'new' + '-repository-' + 'template'})
REWRITTEN_TOKENS = tuple(sorted(_SEEDS)) + tuple(
    '{' + name + '}'
    for name in ('REPOSITORY_NAME', 'DISPLAY_NAME', 'REPOSITORY_OWNER')
)


def test_autonomio_pr_guideline_is_posted_unchanged() -> None:
    """Verify the universal PR guideline exists with the canonical digest."""
    assert PR_GUIDELINE.is_file()
    assert sha256(PR_GUIDELINE.read_bytes()).hexdigest() == EXPECTED_PR_GUIDELINE_SHA256


def test_the_pinned_guideline_carries_no_rewritable_token() -> None:
    """A byte pin is only safe on a file bootstrap leaves alone.

    This is the assertion that would have caught the appendix being pinned. If
    the guideline ever gains the template's own slug or package name, bootstrap
    rewrites it and the digest above fails in every derived repository, inside
    a required check, before the bootstrap PR can merge.
    """
    text = PR_GUIDELINE.read_text(encoding='utf-8')
    for token in REWRITTEN_TOKENS:
        assert token not in text, (
            f'{PR_GUIDELINE.name} contains {token!r}, which bootstrap rewrites. '
            f'A rewritten file cannot carry a byte pin: every derived repository '
            f'would fail this test in a required check.'
        )


def test_repo_specifics_exists_and_points_at_the_guideline() -> None:
    """The appendix must remain an appendix, whatever its entries say."""
    assert REPO_SPECIFICS.is_file()
    assert 'AUTONOMIO_PR_GUIDELINE.md' in REPO_SPECIFICS.read_text(encoding='utf-8')


def test_repo_specifics_carries_scoped_entries() -> None:
    """Its entries stay `[repo:<scope>]`-tagged, which is what makes it usable.

    Checked as shape rather than as bytes: the content is per-repository by
    definition, and pinning those bytes is what broke every derived repository.
    """
    text = REPO_SPECIFICS.read_text(encoding='utf-8')
    assert re.search(r'^- `\[repo:[^\]]+\]`', text, re.MULTILINE), (
        'the appendix carries no `[repo:<scope>]` entries'
    )
