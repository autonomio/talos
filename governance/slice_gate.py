#!/usr/bin/env python3
"""Slice gate -- mechanical enforcement of the PR <-> slice-issue contract.

This gate blocks a PR that:

  1.  Does not cite its slice issue via Closes/Fixes/Resolves #N. The
      closing set must contain exactly one slice-labelled issue; a
      second reference is permitted only for the slice's parent PRD
      under rule 9.
  2.  Cites a number that does not resolve in the repo.
  2a. Cites a number that resolves to a pull request rather than an issue
      (GitHub's issues endpoint also returns PRs).
  3.  Cites an issue that is not in the OPEN state.
  4.  Cites an issue that lacks the ``slice`` label.
  5.  Has a title that does not byte-equal the cited issue's title.
  6.  Cites an issue whose body is missing any full multi-line
      Significance blockquote from the slice template (byte-equal
      substring check).
  7.  Has a diff that touches any file not matched by a glob in the
      cited issue's ``Surfaces`` section.
  8.  Has a diff that touches any file matched by a glob in the cited
      issue's ``Out of Scope`` section.
  9.  Has a closing set that violates PRD closure: while the cited
      slice's parent PRD (native sub-issue parent) has any other open
      sub-issue, the set must be exactly {slice}; when the cited slice
      is the parent's last open sub-issue, exactly {slice, parent PRD};
      a slice with no parent PRD requires exactly {slice}. Every open
      child counts, whatever it is labelled -- a PRD with open children
      is not done.
  10. Cites a slice with a Done Means checkbox neither checked
      (``- [x]``) nor overruled (``OVERRULED: <reason>``). The
      post-merge evidence fields (Merge SHA, Merged PR number, the
      run-id list) are not checkboxes and are exempt pre-merge; they
      are written by ``slice_closeout_guard`` when a merged PR closes
      the issue.

Rules 6 and 7 together bind the template and the issue: the template's
Significance notes cannot drift away from what the validator demands
(rule 6 reads the template at runtime and asserts every blockquote is
in the issue verbatim), and the scope the issue declares cannot be
exceeded by the PR (rule 7 checks the PR diff against the issue's own
Surfaces globs).

Usage:

  python governance/slice_gate.py \
    --pr-title "<pr title>" \
    --pr-body-file <path> \
    --pr-files-file <path>         # one path per line
    --template <path>              # .github/ISSUE_TEMPLATE/slice.yml
    --repo <owner>/<name>

Exit codes:

  0 -- all gates pass
  1 -- one or more gates failed
  2 -- gate itself could not run (bad args, gh failure, etc.)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Final, NoReturn

from _common import CLOSING_KEYWORD_RE, exit_if_bot_exempt, exit_if_disabled, section_setting

# ``##+`` on both the heading and the terminator: issue-form-created
# bodies render field labels as ``###``, and a terminator that only
# recognises ``##`` would silently run the section to end-of-body,
# whitelisting every later bullet for rule 7.
SURFACES_SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r'##+\s+Surfaces\s*\n(.*?)(?=\n##+\s|\Z)',
    re.DOTALL,
)

OUT_OF_SCOPE_SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r'##+\s+Out of Scope\s*\n(.*?)(?=\n##+\s|\Z)',
    re.DOTALL,
)

# Every match is collected and rule 10 requires exactly one section, so
# an appended duplicate (e.g. an all-checked copy after the real one)
# cannot shadow the section that actually carries the boxes. The
# slice_closeout_guard workflow enforces the same exactly-one contract.
DONE_MEANS_SECTION_RE: Final[re.Pattern[str]] = re.compile(
    r'^##+ Done Means\b.*?^##+ Author Checks\b',
    re.MULTILINE | re.DOTALL,
)

# GFM renders -, *, and + bullets (with any indent) as task-list
# checkboxes; matching only ``- [`` would make rule 10 blind to the
# other forms.
CHECKBOX_RE: Final[re.Pattern[str]] = re.compile(
    r'^\s*[-*+] +\[(?P<mark>[ xX])\]\s*(?P<text>.*)$'
)

# Word-bounded so ``NOTOVERRULED:`` does not count, and the literal
# template placeholder ``OVERRULED: <reason>`` is rejected -- an
# overrule needs a real reason, not the copy-pasted example.
OVERRULED_RE: Final[re.Pattern[str]] = re.compile(r'\bOVERRULED:\s*(?!<reason>)\S')

# GitHub also honors closing keywords followed by a qualified
# ``owner/repo#N`` reference or an issue URL. The gate cannot fold
# those into the closing set rules 1 and 9 validate, so their presence
# hard-fails instead of silently widening what a merge will close.
QUALIFIED_CLOSING_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+'
    r'(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#\d+|https?://\S+/issues/\d+)',
    re.IGNORECASE,
)

# The slice issue plus, on the last open slice, its parent PRD (rule 9).
BANNER: Final[str] = 'SLICE GATE'
DEFAULT_MAX_CLOSING_REFERENCES: Final[int] = 2


def _fail_setup(message: str, cause: BaseException | None = None) -> NoReturn:
    """Report a setup failure and exit 2: the gate could not run, which
    is distinct from a rule violation (exit 1). ``raise SystemExit(str)``
    would exit 1 and blur the two."""
    print(message, file=sys.stderr)
    raise SystemExit(2) from cause


def extract_significance_blockquotes(template_path: Path) -> list[str]:
    """Extract every full multi-line Significance blockquote from the
    slice template. Each blockquote is returned as a newline-joined
    string with the ``> `` (or ``>``) prefix preserved, exactly as it
    will appear in an issue body filed via the template.

    This runs at gate time so the validator cannot drift from the
    template: if the template's Significance paragraphs change, the
    gate immediately expects the new text in every slice issue body.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        _fail_setup('slice_gate: PyYAML is required (pip install pyyaml)', exc)
    try:
        with template_path.open(encoding='utf-8') as fh:
            template = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        _fail_setup(
            f'slice_gate: cannot parse template {template_path}: {exc}', exc
        )

    if not isinstance(template, dict):
        _fail_setup(
            f'slice_gate: template {template_path} is not a YAML mapping'
        )

    blocks: list[str] = []
    body_items = template.get('body', [])
    if not isinstance(body_items, list):
        _fail_setup(
            f'slice_gate: template {template_path} has no body list'
        )

    for item in body_items:
        if not isinstance(item, dict):
            continue
        if item.get('type') != 'textarea':
            continue
        attrs = item.get('attributes', {})
        if not isinstance(attrs, dict):
            continue
        value = attrs.get('value')
        if not isinstance(value, str):
            continue
        block = _extract_blockquote_from_value(value)
        if block:
            blocks.append(block)

    if not blocks:
        _fail_setup(
            f'slice_gate: template {template_path} contains no '
            f'Significance blockquotes; cannot run rule 6'
        )
    return blocks


def _extract_blockquote_from_value(value: str) -> str | None:
    """Inside a single textarea's ``value:`` block, pull the contiguous
    blockquote that opens with ``> **Significance.**``. Returns lines
    joined with newlines, verbatim. Returns None if the textarea has
    no such blockquote."""
    collected: list[str] = []
    started = False
    for line in value.split('\n'):
        if not started:
            if line.startswith('> **Significance.**'):
                collected.append(line)
                started = True
        else:
            if line.startswith('>'):
                collected.append(line)
            else:
                break
    return '\n'.join(collected) if collected else None


def _extract_globs_from_section(
    issue_body: str,
    section_pattern: re.Pattern[str],
) -> list[str]:
    """Shared helper: pull every bullet entry from one markdown
    section of the issue body. ``(none)`` is ignored; surrounding
    backticks are stripped."""
    match = section_pattern.search(issue_body)
    if match is None:
        return []
    section = match.group(1)
    globs: list[str] = []
    for raw in section.split('\n'):
        stripped = raw.strip()
        if not stripped.startswith('- '):
            continue
        entry = stripped[2:].strip().strip('`').strip()
        if not entry or entry in {'(none)', 'none'}:
            continue
        globs.append(entry)
    return globs


def extract_surfaces_globs(issue_body: str) -> list[str]:
    """Pull every bullet entry from the ``## Surfaces`` section of the
    issue body. The template's three sub-lists (Modified / Added /
    Removed) are flattened into a single allowed-glob list."""
    return _extract_globs_from_section(issue_body, SURFACES_SECTION_RE)


def extract_out_of_scope_globs(issue_body: str) -> list[str]:
    """Pull every bullet entry from the ``## Out of Scope`` section of
    the issue body. These are the deny-list: any PR file matching one
    of these globs fails the gate, even if the file ALSO matches a
    Surfaces allow-list entry."""
    return _extract_globs_from_section(issue_body, OUT_OF_SCOPE_SECTION_RE)


def _closing_bracket(glob: str, start: int) -> int:
    """Index of the `]` closing the class opened at `start`, or -1.

    Follows `fnmatch`: a leading `!` or `^` negates, and a `]` immediately
    after that is a literal member rather than the terminator.
    """
    index = start + 1
    if index < len(glob) and glob[index] in '!^':
        index += 1
    if index < len(glob) and glob[index] == ']':
        index += 1
    while index < len(glob):
        if glob[index] == ']':
            return index
        index += 1
    return -1


@cache
def _glob_pattern(glob: str, star_crosses_separators: bool = False) -> re.Pattern[str]:
    """Compile one Surfaces glob against a chosen `/` policy.

    With the default policy `/` is a real separator: `*` and `?` stop at one,
    `**` crosses them, and `**/` also matches zero directories so `**/x.py`
    still covers a top-level `x.py`. `fnmatch` has no such policy -- under it
    `*` matches `/` too, which is why `governance/*` silently covered the whole
    subtree beneath it.

    `star_crosses_separators` restores that reach on purpose, for the deny-list
    only. See `path_denied`.
    """
    single_star = '.*' if star_crosses_separators else '[^/]*'
    any_char = '.' if star_crosses_separators else '[^/]'
    out: list[str] = []
    index = 0
    while index < len(glob):
        char = glob[index]
        if char == '*' and glob[index + 1:index + 2] == '*':
            crosses_zero = glob[index + 2:index + 3] == '/'
            out.append('(?:[^/]+/)*' if crosses_zero else '.*')
            index += 3 if crosses_zero else 2
            continue
        if char == '[':
            # A character class, which `fnmatch` honours. Escaping the bracket
            # instead would make `[ab]*.py` a literal that matches no real path
            # -- silently, and on the deny-list that means excluding nothing.
            close = _closing_bracket(glob, index)
            if close != -1:
                body = glob[index + 1:close].replace('\\', '\\\\')
                if body.startswith(('!', '^')):
                    body = '^' + body[1:]
                out.append(f'[{body}]')
                index = close + 1
                continue
        if char == '*':
            out.append(single_star)
        elif char == '?':
            out.append(any_char)
        else:
            out.append(re.escape(char))
        index += 1
    return re.compile(''.join(out) + r'\Z')


def path_matches(path: str, glob: str) -> bool:
    """Whether one repository path is covered by one Surfaces glob."""
    return _glob_pattern(glob).match(path) is not None


def path_denied(path: str, glob: str) -> bool:
    """Whether one path is excluded by one Out of Scope glob.

    The two lists are not symmetric, so they do not share a `/` policy.
    Narrowing the allow-list makes rule 7 stricter; the identical narrowing on
    the deny-list makes rule 8 weaker, and a PR touching an explicitly excluded
    path would pass in silence. Rule 8 therefore keeps the reach `fnmatch` had
    -- a `*` crosses separators here -- which is the fail-closed direction, and
    additionally covers everything beneath a glob that names a directory, so a
    bare `Out of Scope: docs` excludes the tree under it.
    """
    return (
        _glob_pattern(glob, star_crosses_separators=True).match(path) is not None
        or _glob_pattern(
            f"{glob.rstrip('/')}/**", star_crosses_separators=True
        ).match(path) is not None
    )


def read_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        print(
            f'slice_gate: cannot read {path}: {exc}',
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return [line for line in text.splitlines() if line.strip()]


def find_closing_references(body: str) -> list[int]:
    """Return the deduplicated list of issue numbers referenced by
    closing keywords in the PR body, in order of first appearance.
    Deduplication matches GitHub's own closing behavior: repeating
    ``Closes #N`` resolves to one closed issue, not two."""
    return list(dict.fromkeys(
        int(m.group(1)) for m in CLOSING_KEYWORD_RE.finditer(body)
    ))


def fetch_issue(repo: str, number: int) -> dict[str, object] | None:
    """Fetch an issue via gh. Returns None if the issue does not exist.
    Any other gh failure (auth, network, permissions) raises SystemExit(2).

    Implemented with a single subprocess call so a 404 does not emit
    misleading stderr about an unrelated failure path.
    """
    # NOTE: /repos/{repo}/issues/{number} returns pull requests too.
    # Include ``is_pull_request`` in the projection so rule 2a can
    # reject a PR being used as a stand-in for a slice issue.
    try:
        result = subprocess.run(
            [
                'gh', 'api', f'repos/{repo}/issues/{number}',
                '--jq', '{title: .title, state: .state, '
                        'labels: [.labels[].name], body: .body, '
                        'is_pull_request: (.pull_request != null)}',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'slice_gate: gh not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc

    if result.returncode == 0:
        return json.loads(result.stdout) if result.stdout.strip() else None

    # A 404 is the legitimate "issue does not exist" case; every other
    # non-zero exit is a setup failure.
    if 'Not Found' in result.stderr or '404' in result.stderr:
        return None

    print(
        f'slice_gate: gh api repos/{repo}/issues/{number} failed '
        f'(exit {result.returncode}): {result.stderr.strip()}',
        file=sys.stderr,
    )
    raise SystemExit(2)


def fetch_parent_issue_number(repo: str, number: int) -> int | None:
    """Resolve the native sub-issue parent of an issue via the GraphQL
    ``parent`` field (the REST issue object does not expose the
    relationship). Returns None when the issue has no parent. Any gh
    failure raises SystemExit(2)."""
    owner, name = repo.split('/', 1)
    query = (
        'query($owner: String!, $name: String!, $number: Int!) { '
        'repository(owner: $owner, name: $name) { '
        'issue(number: $number) { parent { number } } } }'
    )
    try:
        result = subprocess.run(
            [
                'gh', 'api', 'graphql',
                '-f', f'query={query}',
                '-F', f'owner={owner}',
                '-F', f'name={name}',
                '-F', f'number={number}',
                '--jq', '.data.repository.issue.parent.number',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'slice_gate: gh not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc

    if result.returncode != 0:
        print(
            f'slice_gate: gh api graphql parent lookup for #{number} '
            f'failed (exit {result.returncode}): {result.stderr.strip()}',
            file=sys.stderr,
        )
        raise SystemExit(2)

    stdout = result.stdout.strip()
    return int(stdout) if stdout and stdout != 'null' else None


def fetch_open_sub_issue_numbers(repo: str, parent_number: int) -> list[int]:
    """List the parent PRD's OPEN sub-issue numbers, via the paginated
    native sub-issues API. Any gh failure raises SystemExit(2).

    Every open child counts, not just ``slice``-labelled ones: a PRD with
    open children is not done, whatever they are labelled. Counting only
    slices let rule 9 declare a partially-promoted programme complete --
    PRD #93 had 24 open backlog items and one promoted slice, and the gate
    demanded the PR close the PRD.
    """
    try:
        result = subprocess.run(
            [
                'gh', 'api',
                f'repos/{repo}/issues/{parent_number}/sub_issues',
                '--paginate',
                '--jq', '.[] | select(.state == "open") | .number',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        print(f'slice_gate: gh not found: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc

    if result.returncode != 0:
        print(
            f'slice_gate: gh api repos/{repo}/issues/{parent_number}/'
            f'sub_issues failed (exit {result.returncode}): '
            f'{result.stderr.strip()}',
            file=sys.stderr,
        )
        raise SystemExit(2)

    return [int(line) for line in result.stdout.split()]


def _issue_labels(issue: dict[str, object]) -> list[str]:
    raw_labels = issue.get('labels', [])
    return [str(x) for x in raw_labels] if isinstance(raw_labels, list) else []


def _format_issue_set(numbers: set[int]) -> str:
    return '{' + ', '.join(f'#{n}' for n in sorted(numbers)) + '}'


def _qualified_reference_failures(pr_body: str) -> list[str]:
    """Rule 1's form bound: a closing keyword followed by a qualified
    ``owner/repo#N`` reference or an issue URL is honored by GitHub on
    merge but invisible to the bare ``#N`` parser, so the closing set
    the gate validates would not be the closing set GitHub acts on.
    Such forms fail loud instead of passing unseen."""
    hits = [m.group(0) for m in QUALIFIED_CLOSING_RE.finditer(pr_body)]
    if hits:
        return [
            f'PR body contains {len(hits)} qualified or URL closing '
            f'reference(s) ({", ".join(repr(h) for h in hits)}). GitHub '
            f'honors these on merge but the gate cannot fold them into '
            f'the validated closing set; use the bare `Closes #N` form '
            f'(rule 1).'
        ]
    return []


def _closing_reference_failures(refs: list[int]) -> list[str]:
    """Rule 1's count bounds: zero references, or more than the slice
    plus its parent PRD, fail before any API call."""
    if not refs:
        return [
            'PR body has no closing reference. The closing set must be '
            'exactly the slice issue (`Closes #N`, or Fixes/Resolves, '
            'with N an OPEN slice-labelled issue), plus its parent PRD '
            'only when the slice is the parent\'s last open slice '
            'sub-issue (rule 9).'
        ]
    max_refs = section_setting(
        'slice', 'max_closing_references', DEFAULT_MAX_CLOSING_REFERENCES, BANNER
    )
    if len(refs) > max_refs:
        return [
            f'PR body has {len(refs)} closing references '
            f'({", ".join(f"#{n}" for n in refs)}). The closing set must '
            f'be exactly the slice issue, plus its parent PRD only when '
            f'the slice is the parent\'s last open sub-issue '
            f'(rule 9).'
        ]
    return []


def _fetch_cited_issues(
    repo: str,
    refs: list[int],
) -> tuple[dict[int, dict[str, object]], list[str]]:
    """Rules 2 and 2a per cited number: every reference must resolve to
    a real issue, not a pull request."""
    issues: dict[int, dict[str, object]] = {}
    for number in refs:
        issue_data = fetch_issue(repo, number)
        if issue_data is None:
            return {}, [
                f'issue #{number} does not exist in {repo}. Every '
                f'closing reference must point at a real OPEN issue.'
            ]
        if issue_data.get('is_pull_request'):
            return {}, [
                f'#{number} is a pull request, not an issue. The '
                f'closing reference must point at an OPEN slice issue '
                f'filed via the slice template at '
                f'`.github/ISSUE_TEMPLATE/slice.yml`, not another PR.'
            ]
        issues[number] = issue_data
    return issues, []


def _identify_cited_slice(
    refs: list[int],
    issues: dict[int, dict[str, object]],
) -> tuple[int | None, list[str]]:
    """Identify the cited slice: with one reference it is that issue
    (rule 4 reports a missing slice label); with two, exactly one must
    carry the slice label and the other may only be the parent PRD
    (rule 9 validates the pairing)."""
    slice_refs = [n for n in refs if 'slice' in _issue_labels(issues[n])]
    if len(refs) == 1:
        return refs[0], []
    if len(slice_refs) == 1:
        return slice_refs[0], []
    return None, [
        f'closing references '
        f'({", ".join(f"#{n}" for n in refs)}) contain '
        f'{len(slice_refs)} slice-labelled issues; exactly one must '
        f'be the slice, and the other reference may only be its '
        f'parent PRD (rule 9).'
    ]


def _issue_metadata_failures(
    issue_number: int,
    issue: dict[str, object],
    pr_title: str,
) -> list[str]:
    """Rules 3, 4, and 5: the cited slice is OPEN, slice-labelled, and
    titled byte-identically to the PR."""
    failures: list[str] = []

    state = str(issue.get('state', ''))
    if state.lower() != 'open':
        failures.append(
            f'issue #{issue_number} state is {state!r}; must be OPEN '
            f'for a PR to close it.'
        )

    labels = _issue_labels(issue)
    if 'slice' not in labels:
        failures.append(
            f'issue #{issue_number} is missing the "slice" label '
            f'(labels: {labels!r}). The issue must be filed using the '
            f'slice template at `.github/ISSUE_TEMPLATE/slice.yml`.'
        )

    issue_title = str(issue.get('title', ''))
    if issue_title != pr_title:
        failures.append(
            f'title mismatch: PR title {pr_title!r} does not equal '
            f'issue #{issue_number} title {issue_title!r}. Titles must '
            f'be byte-identical.'
        )

    return failures


def _blockquote_failures(
    issue_number: int,
    issue_body: str,
    template_path: Path,
) -> list[str]:
    """Rule 6: issue body retains every full Significance blockquote
    from the slice template. Blockquotes are extracted from the template
    at runtime so the validator and the template cannot drift apart --
    if a Significance paragraph changes in the template, the new
    paragraph must appear verbatim in every new slice issue's body."""
    required_blockquotes = extract_significance_blockquotes(template_path)
    missing_blocks = [b for b in required_blockquotes if b not in issue_body]
    if not missing_blocks:
        return []
    # Produce a compact failure message: only the first line of each
    # missing blockquote (the heading line) + missing count.
    first_lines = [b.splitlines()[0] for b in missing_blocks]
    return [
        f'issue #{issue_number} body is missing {len(missing_blocks)} '
        f'of {len(required_blockquotes)} full Significance blockquotes '
        f'from the slice template (byte-equal check). Each blockquote '
        f'must appear verbatim in the filed issue. Missing blockquote '
        f'headers: ' + '; '.join(f'{s!r}' for s in first_lines)
    ]


def _scope_failures(
    issue_number: int,
    issue_body: str,
    pr_files: list[str],
) -> list[str]:
    """Rules 7 and 8: the PR diff stays within the issue's Surfaces
    allow-list and touches nothing in its Out of Scope deny-list. A
    file that matches BOTH a Surfaces glob and an Out of Scope glob
    fails rule 8 -- the issue's Out of Scope is the finer-grained
    block."""
    failures: list[str] = []

    allowed_globs = extract_surfaces_globs(issue_body)
    # `**` still matches every path in the repository, and a Surfaces entry
    # that allows everything is not a scope declaration.
    vacuous = sorted(g for g in allowed_globs if g.strip('*/') == '')
    if vacuous:
        failures.append(
            f'issue #{issue_number} Surfaces contains {vacuous!r}, which matches '
            f'every path in the repository. That is not a scope declaration -- '
            f'list the files the slice actually touches.'
        )
    if not allowed_globs:
        failures.append(
            f'issue #{issue_number} Surfaces section has no allowed '
            f'path globs. The scope check requires at least one entry '
            f'under Surfaces (in Modified, Added, or Removed).'
        )
    else:
        not_in_surfaces = [
            f for f in pr_files
            if not any(path_matches(f, g) for g in allowed_globs)
        ]
        if not_in_surfaces:
            failures.append(
                f'PR touches {len(not_in_surfaces)} file(s) not listed in '
                f'issue #{issue_number} Surfaces: '
                + ', '.join(repr(f) for f in not_in_surfaces)
                + f'. Allowed globs: {allowed_globs!r}. '
                  f'Either add the file to the issue\'s Surfaces section '
                  f'(which requires reopening and amending the issue) or '
                  f'remove the change from this PR.'
            )

    denied_globs = extract_out_of_scope_globs(issue_body)
    if denied_globs:
        hits = [
            f for f in pr_files
            if any(path_denied(f, g) for g in denied_globs)
        ]
        if hits:
            failures.append(
                f'PR touches {len(hits)} file(s) listed in issue '
                f'#{issue_number} Out of Scope: '
                + ', '.join(repr(f) for f in hits)
                + f'. Denied globs: {denied_globs!r}. '
                  f'Remove the change from this PR, or amend the issue '
                  f'to move the path out of the Out of Scope section.'
            )

    return failures


def _prd_closure_failures(
    repo: str,
    issue_number: int,
    refs: list[int],
    issues: dict[int, dict[str, object]],
) -> list[str]:
    """Rule 9: the closing set must match the PRD-closure contract. The
    parent PRD is the slice's native sub-issue parent; siblings are the
    parent's other OPEN sub-issues, whatever they are labelled."""
    parent_number = fetch_parent_issue_number(repo, issue_number)
    if parent_number is None:
        expected = {issue_number}
        reason = f'slice #{issue_number} has no parent PRD'
    else:
        open_siblings = [
            n for n in fetch_open_sub_issue_numbers(repo, parent_number)
            if n != issue_number
        ]
        if open_siblings:
            expected = {issue_number}
            reason = (
                f'parent PRD #{parent_number} still has other open '
                f'sub-issues '
                f'({", ".join(f"#{n}" for n in sorted(open_siblings))})'
            )
        else:
            expected = {issue_number, parent_number}
            reason = (
                f'slice #{issue_number} is parent PRD #{parent_number}\'s '
                f'last open sub-issue (no other open children)'
            )
    closing_set = set(refs)
    if closing_set != expected:
        return [
            f'closing set {_format_issue_set(closing_set)} must be '
            f'exactly {_format_issue_set(expected)} because {reason} '
            f'(rule 9).'
        ]
    if parent_number is not None and parent_number in closing_set:
        parent_state = str(issues[parent_number].get('state', ''))
        if parent_state.lower() != 'open':
            return [
                f'parent PRD #{parent_number} state is {parent_state!r}; '
                f'must be OPEN for the PR to close it (rule 9).'
            ]
    return []


def _done_means_failures(issue_number: int, issue_body: str) -> list[str]:
    """Rule 10: every Done Means checkbox in the cited slice is checked
    or explicitly overruled. The post-merge evidence fields (Merge SHA,
    Merged PR number, the run-id list) are not checkboxes, so their
    pre-merge exemption is structural: only checkbox lines are
    inspected. Exactly one Done Means section is required -- an
    appended duplicate could otherwise shadow the real boxes. Silence
    never passes; an overrule needs a reason."""
    sections = list(DONE_MEANS_SECTION_RE.finditer(issue_body))
    if not sections:
        return [
            f'issue #{issue_number} body has no parseable Done Means '
            f'section (## Done Means ... ## Author Checks); rule 10 '
            f'cannot verify checkbox completion.'
        ]
    if len(sections) > 1:
        return [
            f'issue #{issue_number} body has {len(sections)} Done Means '
            f'sections (## Done Means ... ## Author Checks); exactly one '
            f'is required so checkbox completion and closeout evidence '
            f'are unambiguous (rule 10).'
        ]
    dangling: list[str] = []
    for line in sections[-1].group(0).splitlines():
        box = CHECKBOX_RE.match(line)
        if box is None or box.group('mark') != ' ':
            continue
        if OVERRULED_RE.search(box.group('text')):
            continue
        dangling.append(box.group('text').strip() or line.strip())
    if dangling:
        return [
            f'issue #{issue_number} Done Means has {len(dangling)} '
            f'checkbox(es) neither checked nor overruled: '
            + '; '.join(repr(t) for t in dangling)
            + '. Every box must be `- [x]` or carry '
              '`OVERRULED: <reason>` before merge (rule 10).'
        ]
    return []


def gate(
    pr_title: str,
    pr_body: str,
    pr_files: list[str],
    template_path: Path,
    repo: str,
) -> list[str]:
    """Run all PR <-> issue checks. Return a list of failure messages
    (empty list means PASS)."""
    form_failures = _qualified_reference_failures(pr_body)
    if form_failures:
        return form_failures

    refs = find_closing_references(pr_body)

    count_failures = _closing_reference_failures(refs)
    if count_failures:
        return count_failures

    issues, fetch_failures = _fetch_cited_issues(repo, refs)
    if fetch_failures:
        return fetch_failures

    issue_number, pairing_failures = _identify_cited_slice(refs, issues)
    if issue_number is None:
        return pairing_failures
    issue = issues[issue_number]

    # CRLF is normalised before the body-content rules: a body saved
    # through the web editor arrives with \r\n and would otherwise fail
    # every byte-equal substring check.
    issue_body = str(issue.get('body') or '').replace('\r\n', '\n')

    metadata_failures = _issue_metadata_failures(issue_number, issue, pr_title)
    failures: list[str] = list(metadata_failures)
    failures.extend(_blockquote_failures(issue_number, issue_body, template_path))
    failures.extend(_scope_failures(issue_number, issue_body, pr_files))
    # Rule 9's graph lookups are meaningful only for an OPEN,
    # slice-labelled, correctly-titled citation; when rules 3-5 already
    # failed, skipping them keeps a plain rule violation from turning
    # into an exit-2 API failure that would obscure it.
    if not metadata_failures:
        failures.extend(_prd_closure_failures(repo, issue_number, refs, issues))
    failures.extend(_done_means_failures(issue_number, issue_body))
    return failures


def main() -> int:
    exit_if_disabled('slice', BANNER)
    parser = argparse.ArgumentParser(description='Slice gate')
    parser.add_argument(
        '--pr-title',
        required=True,
        help='The PR title, byte-equal to what will match against the issue.',
    )
    parser.add_argument(
        '--pr-body-file',
        required=True,
        help='Path to a UTF-8 file containing the PR body verbatim.',
    )
    parser.add_argument(
        '--pr-files-file',
        required=True,
        help='Path to a UTF-8 file with one changed file path per line '
             '(as produced by `gh pr diff --name-only`).',
    )
    parser.add_argument(
        '--template',
        required=True,
        help='Path to the slice template '
             '(e.g. .github/ISSUE_TEMPLATE/slice.yml). The validator '
             'loads the template at runtime and requires every '
             'Significance blockquote to be present verbatim in the '
             'cited issue body.',
    )
    parser.add_argument(
        '--repo',
        required=True,
        help='GitHub repository in owner/name form (e.g. autonomio/talos).',
    )
    parser.add_argument(
        '--pr-author',
        default='',
        help='Login of the account that opened the PR. When it appears in '
             '`automation.bot_authors` and this gate is in '
             '`automation.exempt_gates`, the gate skips. Empty means enforce.',
    )
    args = parser.parse_args()
    exit_if_bot_exempt('slice', args.pr_author, BANNER)

    try:
        pr_body = Path(args.pr_body_file).read_text(encoding='utf-8')
    except OSError as exc:
        print(
            f'slice_gate: cannot read --pr-body-file {args.pr_body_file}: {exc}',
            file=sys.stderr,
        )
        return 2

    pr_files = read_lines(Path(args.pr_files_file))

    template_path = Path(args.template)
    if not template_path.is_file():
        print(
            f'slice_gate: --template {template_path} does not exist',
            file=sys.stderr,
        )
        return 2

    failures = gate(
        args.pr_title,
        pr_body,
        pr_files,
        template_path,
        args.repo,
    )

    if failures:
        print('SLICE GATE -- FAIL')
        print()
        for msg in failures:
            print(f'  - {msg}')
        print()
        print(f'{len(failures)} failure(s). Merge blocked.')
        return 1

    print('SLICE GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
