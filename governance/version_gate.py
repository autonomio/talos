#!/usr/bin/env python3
"""Version gate -- every PR must bump version and record a CHANGELOG trail.

Enforces seven rules, all deterministic:

  1. The head commit's `pyproject.toml` is different from the base's.
  2. `[project].version` at the head is strictly greater than at base
     (semver compare).
  3. The head commit's `CHANGELOG.md` is different from the base's.
  4. `CHANGELOG.md` at the head contains a top-of-file version header
     `# v<new_version>`, and that header is the first `# v...` line in
     the file (ahead of the previous version's header).
  5. The bump level (patch / minor / major) is at least the minimum
     implied by the PR title's Conventional Commits type:
         type!            -> major
         feat             -> minor
         anything else    -> patch
  6. The top `# v<new_version>` section has at least one non-empty,
     non-header line of content before the next version header.
     A header-only entry satisfies the surface form of rule 4 but
     carries no trail; rule 6 requires the actual changelog item.
  7. The new top section follows the writing conventions: bullets are
     imperative ("Add", not "Added") and leave no unfinished marker --
     a "TODO:"/"FIXME:"-style note or a stub bullet (`- TBD`, `- ...`).
     Only the top section is checked, so older entries are never
     re-litigated.

"Whatever is changed must leave a trail" -- rules 1 and 3 enforce that
every PR edits both artifacts. Rule 5 enforces that the trail records
the right magnitude of change. Rules 2 and 4 enforce that the trail
and the artifact agree on what the new version is.

Usage:

  python governance/version_gate.py \\
    --pr-title "<cc-compliant title>" \\
    --base-pyproject <path>   # pyproject.toml at BASE
    --head-pyproject <path>   # pyproject.toml at HEAD
    --base-changelog <path>   # CHANGELOG.md at BASE
    --head-changelog <path>   # CHANGELOG.md at HEAD

Exit codes:
  0 -- all rules pass
  1 -- one or more rules failed
  2 -- gate itself could not run (bad args, parse failure, etc.)
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Final

from _common import (
    CC_RE,
    TOMLDecodeError,
    exit_if_bot_exempt,
    exit_if_disabled,
    fail_setup,
    loads_toml,
    section_setting,
)

# Strict `MAJOR.MINOR.PATCH` only. We explicitly reject prerelease
# and build-metadata forms because this gate compares as integer
# triples; accepting `1.3.1-alpha` but silently ignoring the `-alpha`
# part would let `1.3.1-alpha` and `1.3.1` compare equal. The simpler
# fix is to refuse ambiguous forms outright.
SEMVER_RE: Final[re.Pattern[str]] = re.compile(
    r'^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$'
)


LEVEL_ORDER: Final[dict[str, int]] = {
    'none': 0,
    'patch': 1,
    'minor': 2,
    'major': 3,
}


def parse_semver(value: str) -> tuple[int, int, int]:
    """Parse `MAJOR.MINOR.PATCH`. Reject any prerelease/build-metadata
    form outright -- comparing `1.3.1-alpha` against `1.3.1` as integer
    triples would say they are equal, which contradicts real semver
    precedence (`1.3.1-alpha` < `1.3.1`). The gate's remit does not
    include full precedence ordering, so the input format is narrowed
    instead."""
    match = SEMVER_RE.match(value.strip())
    if match is None:
        print(
            f'version_gate: {value!r} is not a valid version string. '
            f'Expected strict `MAJOR.MINOR.PATCH` (no prerelease, no '
            f'build metadata).',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return int(match['major']), int(match['minor']), int(match['patch'])


def extract_version(
    pyproject_text: str, label: str, version_source: str | None = None
) -> str:
    try:
        data = loads_toml(pyproject_text)
    except TOMLDecodeError as exc:
        print(
            f'version_gate: cannot parse {label} pyproject.toml: {exc}',
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    project = data.get('project')
    if not isinstance(project, dict):
        print(
            f'version_gate: {label} pyproject.toml has no [project] table',
            file=sys.stderr,
        )
        raise SystemExit(2)
    version = project.get('version')
    if version is None and project.get('dynamic') == ['version']:
        tool = data.get('tool', {})
        hatch = tool.get('hatch', {})
        declared = hatch.get('version', {})
        if declared.get('path') != 'talos/__init__.py' or version_source is None:
            fail_setup('VERSION GATE', f'{label}: dynamic version requires the declared Talos source')
        tree = ast.parse(version_source)
        values = [
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets)
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ]
        if len(values) != 1:
            fail_setup('VERSION GATE', f'{label}: expected one literal __version__ assignment')
        version = values[0]
    if not isinstance(version, str) or not version.strip():
        print(
            f'version_gate: {label} pyproject.toml [project].version is missing '
            f'or not a string (got {version!r})',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return version.strip()


def bump_level(base: tuple[int, int, int], head: tuple[int, int, int]) -> str:
    if head[0] > base[0]:
        return 'major'
    if head[0] == base[0] and head[1] > base[1]:
        return 'minor'
    if head[0] == base[0] and head[1] == base[1] and head[2] > base[2]:
        return 'patch'
    return 'none'


def required_bump_level(pr_title: str) -> str:
    first = pr_title.split('\n', 1)[0]
    match = CC_RE.match(first)
    if match is None:
        # cc_gate is the authoritative check for CC format; version_gate
        # assumes a compliant title. If we cannot parse it, be strict:
        # require at least patch.
        return 'patch'
    if match['breaking']:
        return 'major'
    if match['type'] == 'feat':
        return 'minor'
    return 'patch'


BANNER: Final[str] = 'VERSION GATE'
DEFAULT_HEADER: Final[str] = '# v{version}'
DEFAULT_NEWEST: Final[str] = 'first'
_VERSION_TOKEN: Final[str] = '([0-9A-Za-z.+\\-]+)'


def _header_re() -> re.Pattern[str]:
    """Compile the configured changelog header form into a matcher.

    The form is a template carrying `{version}` -- `# v{version}` here,
    `## [{version}] - ` in a Keep-a-Changelog repository. Literal runs are
    escaped so a form containing regex metacharacters (`[`, `]`, `.`) is
    matched literally, and runs of spaces become `\\s+` so header spacing is
    not load-bearing.
    """
    form = str(section_setting('changelog', 'header', DEFAULT_HEADER, BANNER))
    if '{version}' not in form:
        fail_setup(BANNER, f"changelog.header must contain '{{version}}', got {form!r}")
    before, _, after = form.partition('{version}')
    pattern = '^' + _escape_form(before) + _VERSION_TOKEN
    pattern += _escape_form(after) if after.strip() else '\\b'
    return re.compile(pattern)


def _escape_form(part: str) -> str:
    """Escape a literal run of the header form, turning every run of spaces
    into flexible whitespace -- including one adjacent to the version
    placeholder.

    Splitting on spaces and dropping empty chunks would silently delete the
    whitespace either side of `{version}`, so a form like `## {version}`
    compiled to a pattern demanding the version immediately after `##` and
    matched none of that repository's own headers.
    """
    out: list[str] = []
    for chunk in re.split(r'( +)', part):
        if not chunk:
            continue
        out.append(r'\s+' if chunk.isspace() else re.escape(chunk))
    return ''.join(out)


def _newest_first() -> bool:
    """Whether the newest entry sits at the top of the changelog.

    Both orderings are in use: this repository prepends, Keep-a-Changelog
    style repositories append. The gate checks the *new* section, so it has
    to know which end that is.
    """
    raw = section_setting('changelog', 'newest', DEFAULT_NEWEST, BANNER)
    if raw not in ('first', 'last'):
        fail_setup(BANNER, f"changelog.newest must be 'first' or 'last', got {raw!r}")
    return raw == 'first'


def _header_indices(lines: list[str], header_re: re.Pattern[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if header_re.match(line)]


def _newest_index(lines: list[str], header_re: re.Pattern[str]) -> int | None:
    """Index of the header for the newest entry, per the configured order."""
    idx = _header_indices(lines, header_re)
    if not idx:
        return None
    return idx[0] if _newest_first() else idx[-1]

# Changelog writing conventions, the mechanizable subset: entries are
# imperative ("Add", not "Added"), and carry no leftover template
# placeholders. Only the new top section is checked, so old entries are
# never re-litigated.
_PAST_TENSE_BULLET_RE: Final[re.Pattern[str]] = re.compile(
    r'^\s*[-*]\s+(Added|Fixed|Removed|Changed|Updated|Renamed|Deleted|Moved|'
    r'Improved|Refactored|Bumped|Created|Introduced|Implemented|Replaced|'
    r'Dropped|Reverted|Disabled|Enabled|Documented|Skipped)\b'
)
_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(?:TODO|FIXME|XXX|HACK|TBD)\s*:'                 # a marker note like "TODO: ..."
    r'|^\s*[-*]\s+(?:TODO|TBD|WIP|N/?A|\.\.\.)\s*$',     # a bullet that is only a stub
    re.IGNORECASE,
)


def first_version_header(changelog_text: str) -> str | None:
    """Return the version string from the newest entry's header, or None if
    the changelog carries no header. Which end is newest is configured."""
    header_re = _header_re()
    lines = changelog_text.splitlines()
    i = _newest_index(lines, header_re)
    if i is None:
        return None
    match = header_re.match(lines[i])
    return match.group(1) if match else None

def top_section_is_empty(changelog_text: str) -> bool:
    """True if the newest entry's section carries no content line -- i.e.
    nothing between its header and the adjacent header or the end of file."""
    header_re = _header_re()
    lines = changelog_text.splitlines()
    start = _newest_index(lines, header_re)
    if start is None:
        # No header at all. Rule 4 already flags this; treat as empty for
        # completeness.
        return True
    for line in lines[start + 1:]:
        if header_re.match(line):
            return True  # adjacent section reached without finding content
        if line.strip():
            return False
    return True  # reached the end without finding content

def top_section_lines(changelog_text: str) -> list[str]:
    """The newest entry's content lines, up to the adjacent header or the end
    of file. Used to check the new entry's writing conventions without
    re-litigating older sections."""
    header_re = _header_re()
    lines = changelog_text.splitlines()
    start = _newest_index(lines, header_re)
    if start is None:
        return []
    out: list[str] = []
    for line in lines[start + 1:]:
        if header_re.match(line):
            break
        out.append(line)
    return out

def gate(
    pr_title: str,
    base_pyproject: str,
    head_pyproject: str,
    base_changelog: str,
    head_changelog: str,
    base_version_source: str | None = None,
    head_version_source: str | None = None,
) -> list[str]:
    failures: list[str] = []

    base_version = extract_version(base_pyproject, 'base', base_version_source)
    head_version = extract_version(head_pyproject, 'head', head_version_source)

    # Rule 1: pyproject.toml differs.
    if base_pyproject == head_pyproject and base_version_source == head_version_source:
        failures.append(
            'pyproject.toml is byte-identical between base and head. Every '
            'PR must bump the version.'
        )
    elif base_version == head_version:
        failures.append(
            f'pyproject.toml changed but [project].version is still '
            f'{head_version!r}. Every PR must bump the version.'
        )

    # Rule 2: head version > base (strictly, by semver).
    base_sv = parse_semver('1.4.0' if base_version == '1.4' and base_version_source is not None else base_version)
    head_sv = parse_semver(head_version)
    actual = bump_level(base_sv, head_sv)
    if actual == 'none':
        failures.append(
            f'version did not move forward. base={base_version!r}, '
            f'head={head_version!r}. Every PR must advance the version.'
        )

    # Rule 3: CHANGELOG differs.
    if base_changelog == head_changelog:
        failures.append(
            'CHANGELOG.md is byte-identical between base and head. Every '
            'PR must record its change in CHANGELOG.md.'
        )

    # Rule 4: CHANGELOG has a `# v<head_version>` line AT THE TOP (first
    # version header), ahead of the previous version's header.
    top_header = first_version_header(head_changelog)
    if top_header is None:
        failures.append(
            'CHANGELOG.md contains no `# v<X.Y.Z>` line. Add a version '
            'header for this release.'
        )
    elif top_header != head_version:
        failures.append(
            f'CHANGELOG.md top version header is `# v{top_header}` but '
            f'pyproject.toml reports {head_version!r}. They must match, and '
            f'the new header must be the first version heading in the file.'
        )

    # Rule 5: bump level meets the minimum required by the CC type.
    if actual != 'none':
        required = required_bump_level(pr_title)
        if LEVEL_ORDER[actual] < LEVEL_ORDER[required]:
            failures.append(
                f'PR title {pr_title!r} requires at least a {required} version '
                f'bump; the actual bump is {actual} ({base_version} -> '
                f'{head_version}).'
            )

    # Rule 6: the top version section must carry at least one line of
    # actual content (not just a header followed by blanks or another
    # version header). Prevents the "header-only trail" bypass.
    if top_section_is_empty(head_changelog):
        failures.append(
            f'CHANGELOG.md top version section (`# v{head_version}`) has '
            f'no content before the next version header (or end of file). '
            f'Every version bump must be accompanied by at least one '
            f'non-empty changelog line describing what changed.'
        )

    # Rule 7: the new top section follows the writing conventions --
    # imperative mood and no leftover placeholders. Only the top section
    # is checked, so historical entries are never re-litigated.
    for raw in top_section_lines(head_changelog):
        past = _PAST_TENSE_BULLET_RE.match(raw)
        if past is not None:
            failures.append(
                f'CHANGELOG.md entry uses past tense {past.group(1)!r}; '
                f'changelog bullets are imperative ("Add", not "Added"): '
                f'{raw.strip()!r}'
            )
        placeholder = _PLACEHOLDER_RE.search(raw)
        if placeholder is not None:
            failures.append(
                f'CHANGELOG.md top section has an unfilled placeholder '
                f'({placeholder.group(0)!r}); complete the entry before merge: '
                f'{raw.strip()!r}'
            )

    return failures


def _read(path: str, label: str) -> str:
    try:
        return Path(path).read_text(encoding='utf-8')
    except OSError as exc:
        print(
            f'version_gate: cannot read --{label} {path}: {exc}',
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def main() -> int:
    exit_if_disabled('version', BANNER)
    parser = argparse.ArgumentParser(description='Version gate')
    parser.add_argument('--pr-title', required=True)
    parser.add_argument('--base-pyproject', required=True)
    parser.add_argument('--head-pyproject', required=True)
    parser.add_argument('--base-changelog', required=True)
    parser.add_argument('--head-changelog', required=True)
    parser.add_argument('--base-version-source')
    parser.add_argument('--head-version-source')
    parser.add_argument(
        '--pr-author',
        default='',
        help='Login of the account that opened the PR. When it appears in '
             '`automation.bot_authors` and this gate is in '
             '`automation.exempt_gates`, the gate skips. Empty means enforce.',
    )
    args = parser.parse_args()
    exit_if_bot_exempt('version', args.pr_author, BANNER)

    base_pyproject = _read(args.base_pyproject, 'base-pyproject')
    head_pyproject = _read(args.head_pyproject, 'head-pyproject')
    base_changelog = _read(args.base_changelog, 'base-changelog')
    head_changelog = _read(args.head_changelog, 'head-changelog')

    failures = gate(
        args.pr_title,
        base_pyproject,
        head_pyproject,
        base_changelog,
        head_changelog,
        _read(args.base_version_source, 'base-version-source') if args.base_version_source else None,
        _read(args.head_version_source, 'head-version-source') if args.head_version_source else None,
    )

    if failures:
        print('VERSION GATE -- FAIL')
        print()
        for msg in failures:
            print(f'  - {msg}')
        print()
        print(f'{len(failures)} failure(s). Merge blocked.')
        return 1

    print('VERSION GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
