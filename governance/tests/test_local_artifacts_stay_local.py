"""Prove a tool's scratch output cannot be committed by accident.

`uv run --with <tool>` writes a `uv.lock` for the ad-hoc environment it builds.
This project does not resolve through uv -- CI installs the compiled,
hash-pinned sets under `requirements/ci/` -- so such a file is a local artifact
and not a dependency source. One was committed by a `git add -A`, where it
broke the manifest check and put a file no gate governs inside a slice's diff.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _in_git_work_tree() -> bool:
    """Whether this tree is a git work tree, distinguishing "no" from "cannot tell".

    Collapsing every git failure into `False` would let a dubious-ownership
    refusal, or an unreadable repository, silently skip all three checks in a
    repository where they were meant to run. Only the answer git actually gives
    -- no repository here -- is allowed to skip; anything else is raised, which
    fails collection loudly rather than passing quietly.
    """
    result = subprocess.run(
        ['git', 'rev-parse', '--is-inside-work-tree'],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip() == 'true'
    if 'not a git repository' in result.stderr.lower():
        return False
    raise RuntimeError(
        f'git could not determine whether {REPO_ROOT} is a work tree '
        f'(exit {result.returncode}): {result.stderr.strip()}'
    )


# A derived repository is a plain directory copy until someone runs `git init`,
# and `git check-ignore` has no answer outside a work tree. Skipping there is
# the honest reading -- the question does not apply -- and it stays loud,
# because in this repository the condition is never true and the tests always
# run. Asserting instead would ship a failing suite to every derived repository.
pytestmark = pytest.mark.skipif(
    not _in_git_work_tree(),
    reason='not a git work tree; `git check-ignore` cannot answer here',
)


def _is_ignored(relative_path: str) -> bool:
    """Ask git itself, rather than pattern-matching `.gitignore` by hand.

    A test that only asserted the literal line was present would pass while a
    later `!uv.lock` negation, or an ordering change, silently re-exposed the
    file.
    """
    result = subprocess.run(
        ['git', 'check-ignore', '-q', relative_path],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def test_a_uv_lockfile_cannot_be_committed() -> None:
    assert _is_ignored('uv.lock'), (
        'uv.lock is not ignored. `uv run --with <tool>` writes one, and a '
        '`git add -A` then sweeps it into the commit, where it breaks the '
        'manifest check and lands outside the slice Surfaces.'
    )


def test_the_environments_that_produce_it_are_ignored_too() -> None:
    """The lockfile is one of a family; the venvs beside it stay local as well.

    Probed through a path *inside* each directory rather than the bare name. A
    trailing-slash pattern is directory-only, and `git check-ignore` matches one
    only once the directory exists, so asserting on `venv` alone would go red on
    any checkout that has not created it yet -- while the rule was intact.
    """
    for path in (
        '.venv/pyvenv.cfg',
        'venv/lib/python3.12/site-packages/x.py',
        '.ruff_cache/content.json',
        'governance/__pycache__/slice_gate.pyc',
    ):
        assert _is_ignored(path), f'{path} is not ignored'


def test_no_local_artifact_is_tracked_right_now() -> None:
    """Ignoring a path does nothing once the file is already tracked."""
    tracked = subprocess.run(
        ['git', 'ls-files', 'uv.lock', '.venv', '.ruff_cache', 'venv'],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert not tracked, f'local artifacts are tracked despite being ignored: {tracked}'
