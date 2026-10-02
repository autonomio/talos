#!/usr/bin/env python3
"""Shared helpers for the governance gates.

One place for the things several gates would otherwise each re-implement:
the repo root, setup-failure reporting, package-root resolution, the
significant-line counter, and the Conventional-Commits patterns. This is
infrastructure shared by the gates, not a gate itself — it has no banner and
is never a required check. Sharing a constant here (e.g. `CC_RE`) is not the
same as coupling two gates' logic; the gates remain independently invocable.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Final, NoReturn, TypeVar, cast

# `tomllib` is stdlib only from 3.11. Guarded once, here, rather than at
# each of the six call sites: a derived repository with a lower floor needs
# the fallback, and duplicating the guard put a `try`/`except` under
# `tests/` where the test-fallback gate reads it as a swallowed assertion.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

T = TypeVar('T')

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

# The Conventional Commits subject regex: type, optional scope (without
# parens), optional breaking marker, description. Shared by cc_gate and
# version_gate so the two cannot silently drift apart.
CC_RE: Final[re.Pattern[str]] = re.compile(
    r'^(?P<type>[a-z]+)'
    r'(?:\((?P<scope>[a-z0-9._/\-]+)\))?'
    r'(?P<breaking>!)?'
    r': (?P<description>.+)$'
)

# Issue-closing keyword regex, shared by cc_gate and slice_gate.
CLOSING_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(
    r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b',
    re.IGNORECASE,
)


def fail_setup(banner: str, message: str) -> NoReturn:
    """Report a gate setup failure under the gate's banner and exit 2.

    Setup failures (a missing config, an unresolvable package root) are
    distinct from gate violations: they mean the gate could not run, so it
    fails closed rather than passing over an empty tree.
    """
    print(f'{banner} -- FAIL', file=sys.stderr)
    print(f'  {message}', file=sys.stderr)
    sys.exit(2)


GOVERNANCE_CONFIG: Final[Path] = REPO_ROOT / 'governance.yml'
BUDGETS: Final[Path] = REPO_ROOT / '.github' / 'budgets.json'


def config(banner: str = 'CONFIG') -> dict[str, Any]:
    """Read `governance.yml`, the repository's configuration.

    Carries per-repository shape -- what the repository is, where things live,
    which gates run, and the policy numbers they enforce. Distinct from
    `.github/budgets.json`, which carries ratchets: values that may only move
    one way.

    Absent and malformed are deliberately different. An absent file means the
    repository configured nothing, so every gate enforces the default it
    documents -- which is what lets a repository adopt one gate without
    authoring a whole config. A malformed file means the repository tried to
    say something the gate cannot read, and guessing there would enforce
    something nobody asked for, so it blocks.

    The values with no honest default -- `layout.package_root` above all --
    are not covered by this. `resolve_package_dir` blocks on its own when the
    package root is missing, so an absent config cannot silently point a gate
    at nothing.
    """
    if not GOVERNANCE_CONFIG.is_file():
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - gate-tools always installs it
        fail_setup(banner, f'PyYAML is required to read governance.yml: {exc}')
    try:
        raw = yaml.safe_load(GOVERNANCE_CONFIG.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        fail_setup(banner, f'cannot read {GOVERNANCE_CONFIG.relative_to(REPO_ROOT)}: {exc}')
    if not isinstance(raw, dict):
        fail_setup(banner, f'{GOVERNANCE_CONFIG.relative_to(REPO_ROOT)} is not a mapping')
    return raw


def gate_config(name: str, banner: str) -> dict[str, Any]:
    """Read one gate's section from `gates`, failing closed on a bad shape.

    A missing section is an empty mapping, so a gate keeps the defaults it
    documents and a repository configures only what it needs to change.
    """
    gates = config(banner).get('gates', {})
    if not isinstance(gates, dict):
        fail_setup(banner, 'governance.yml: `gates` must be a mapping')
    value = gates.get(name, {})
    if not isinstance(value, dict):
        fail_setup(banner, f'governance.yml: gates.{name} must be a mapping')
    return value


def gate_enabled(name: str, banner: str = 'CONFIG') -> bool:
    """Whether a gate is switched on. Absent means on.

    Validated rather than truthy-tested: `enabled: no` is a string in YAML's
    eyes under some quoting, and an unvalidated read would treat any typo as
    "on" -- or, with the opposite test, silently disable a gate. Both are the
    failure this switch exists to make visible, so a non-boolean blocks.
    """
    return _validated(
        f'gates.{name}', 'enabled', gate_config(name, banner).get('enabled', True), True, banner
    )


def _validated(where: str, key: str, value: object, default: T, banner: str) -> T:
    """Validate one configured value against the shape of its default."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            fail_setup(banner, f'{where}.{key} must be a boolean, got {value!r}')
        return cast('T', value)
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail_setup(banner, f'{where}.{key} must be a number, got {value!r}')
        if default > 0 and value <= 0:
            fail_setup(banner, f'{where}.{key} must be positive, got {value!r}')
        return cast('T', type(default)(value))
    if isinstance(default, (list, tuple, frozenset, set)):
        if not isinstance(value, (list, tuple)) or not all(isinstance(e, str) for e in value):
            fail_setup(banner, f'{where}.{key} must be a list of strings, got {value!r}')
        return cast('T', type(default)(value))
    if not isinstance(value, type(default)):
        fail_setup(banner, f'{where}.{key} must be {type(default).__name__}, got {value!r}')
    return cast('T', value)


def exit_if_disabled(name: str, banner: str) -> None:
    """Leave the gate immediately when the repository switched it off.

    One line at each call site rather than three, because the gate scripts are
    held to a 120-line self-limit and a switch is not worth 3% of that budget
    in every one of them.
    """
    if not gate_enabled(name, banner):
        print(f'{banner} -- SKIP (gates.{name}.enabled is false)')
        raise SystemExit(0)


def exit_if_bot_exempt(name: str, author: str, banner: str) -> None:
    """Leave the gate when a listed bot authored the pull request.

    Both lists default to empty, so a repository that never configures
    `automation` enforces every law on every author. The skip is announced with
    the author and the gate that let it through: an exemption nobody can see in
    the log is one nobody audits.
    """
    if not author:
        return
    authors = section_setting('automation', 'bot_authors', [], banner)
    exempt = section_setting('automation', 'exempt_gates', [], banner)
    if author in authors and name in exempt:
        print(f'{banner} -- SKIP (automation.exempt_gates lists {name} for bot {author})')
        raise SystemExit(0)


def gate_setting(gate: str, key: str, default: T, banner: str) -> T:
    """Read one typed setting from a gate's section, failing closed.

    Fails on a value of the wrong type or a non-positive number where the
    default is positive: a gate cannot check against a bound it cannot parse,
    and silently substituting the default would enforce something the
    repository did not ask for.
    """
    return _validated(
        f'gates.{gate}', key, gate_config(gate, banner).get(key, default), default, banner
    )


def section_setting(name: str, key: str, default: T, banner: str) -> T:
    """Read one typed setting from a top-level section.

    Distinct from `gate_setting` because `slice` and `changelog` are not
    gates. Reading them through the gate reader looked for `gates.slice` and
    `gates.changelog`, which do not exist, so the declared values were inert
    and every gate silently used its own default.
    """
    return _validated(
        name, key, section(name, banner).get(key, default), default, banner
    )


def section(name: str, banner: str = 'CONFIG') -> dict[str, Any]:
    """Read one top-level section, failing closed when it is not a mapping."""
    value = config(banner).get(name, {})
    if not isinstance(value, dict):
        fail_setup(banner, f'governance.yml: `{name}` must be a mapping')
    return value


def budgets(banner: str) -> dict[str, Any]:
    """Read `.github/budgets.json`, the ratcheting values, failing closed."""
    if not BUDGETS.is_file():
        fail_setup(banner, f'missing {BUDGETS.relative_to(REPO_ROOT)}')
    try:
        raw = json.loads(BUDGETS.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        fail_setup(banner, f'cannot read {BUDGETS.relative_to(REPO_ROOT)}: {exc}')
    if not isinstance(raw, dict):
        fail_setup(banner, f'{BUDGETS.relative_to(REPO_ROOT)} is not a JSON object')
    return raw


def write_budget_section(section_name: str, payload: dict[str, Any], banner: str) -> Path:
    """Merge one section into `.github/budgets.json`, preserving the others.

    The budgets were one file per gate before they were merged; each gate's
    `--update-budget` wrote its whole dict back, which is still correct for a
    file it owns alone and destroys five ratchets in a file it shares. The
    merge happens here so neither gate can get it wrong independently.
    """
    existing: dict[str, Any] = {}
    if BUDGETS.is_file():
        try:
            loaded = json.loads(BUDGETS.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            fail_setup(banner, f'cannot read {BUDGETS.relative_to(REPO_ROOT)}: {exc}')
        if not isinstance(loaded, dict):
            fail_setup(banner, f'{BUDGETS.relative_to(REPO_ROOT)} is not a JSON object')
        existing = loaded
    existing[section_name] = payload
    BUDGETS.parent.mkdir(parents=True, exist_ok=True)
    BUDGETS.write_text(json.dumps(existing, indent=2) + '\n', encoding='utf-8')
    return BUDGETS


def resolve_package_dir(banner: str) -> Path:
    """Resolve the package directory from `layout.package_root`.

    A gate that cannot find its scan target must block the merge instead of
    passing over an empty tree, so a half-finished package rename cannot
    silently disable it.
    """
    root = section('layout', banner).get('package_root')
    path = REPO_ROOT / root if isinstance(root, str) and root else None
    if path is None or not path.is_dir():
        fail_setup(banner, f'layout.package_root {root!r} is not a directory under the repo root')
    return path


def resolve_paths(key: str, banner: str) -> list[Path]:
    """Resolve one `layout` path list, keeping only the entries that exist.

    A configured path that is absent is not an error: a repository may declare
    `gate_test_paths` it has not created yet. A configured list that is not a
    list of strings is an error, because the gate cannot tell what to scan.
    """
    raw = section('layout', banner).get(key, [])
    if not isinstance(raw, list) or not all(isinstance(entry, str) for entry in raw):
        fail_setup(banner, f'layout.{key} must be a list of strings, got {raw!r}')
    return sorted({path for entry in raw for path in REPO_ROOT.glob(entry) if path.exists()})


def layout_excludes(gate: str, banner: str) -> list[str]:
    """The repo-wide excludes plus the ones this gate adds, validated.

    Extending rather than replacing means a per-gate list cannot re-admit
    build output that the repository declared out of scope for everything.
    """
    merged: list[str] = []
    for where, raw in (
        ('layout.excludes', section('layout', banner).get('excludes', [])),
        (f'gates.{gate}.excludes', gate_config(gate, banner).get('excludes', [])),
    ):
        if not isinstance(raw, list) or not all(isinstance(entry, str) for entry in raw):
            fail_setup(banner, f'{where} must be a list of strings, got {raw!r}')
        merged.extend(entry for entry in raw if entry not in merged)
    return merged


def scan_surface_failures(
    base_config_path: str | None, banner: str, gate: str | None = None
) -> list[str]:
    """Report any way the PR narrows the tree its own ratchets are measured over.

    `package_root` and `excludes` live in `governance.yml`, so a PR editing
    that file could point a ratchet at a smaller subtree, or exclude the very
    files carrying its new escape hatches, and pass on a count taken over
    less code than the base ref was measured against. Both ratcheting gates
    call this, so the two cannot drift apart.

    `gate` names the calling gate so the comparison covers the same excludes
    the gate actually scans. A gate resolves its tree through
    `layout_excludes`, which merges `gates.<gate>.excludes` into the repo-wide
    list; comparing only the repo-wide half left the per-gate half as an
    unguarded lever, which is the exact defect this function exists to
    prevent. Passing None compares the repo-wide list alone.
    """
    if base_config_path is None:
        return []
    base_path = Path(base_config_path)
    if not base_path.is_file():
        return [
            f'{banner}: base-ref governance.yml not found at {base_path}. The scan '
            f'surface cannot be compared, so the ratchet cannot be trusted. '
            f'Restore it on the base ref.'
        ]
    import yaml  # type: ignore[import-untyped]

    try:
        base_raw = yaml.safe_load(base_path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        return [f'{banner}: cannot read base governance.yml {base_path}: {exc}']
    if not isinstance(base_raw, dict):
        return [f'{banner}: base governance.yml {base_path} is not a mapping']
    base_layout = base_raw.get('layout', {})
    if not isinstance(base_layout, dict):
        return [f'{banner}: base governance.yml {base_path} has a non-mapping `layout`']

    failures: list[str] = []
    head_layout = section('layout', banner)
    base_root = base_layout.get('package_root')
    head_root = head_layout.get('package_root')
    if base_root != head_root:
        failures.append(
            f'layout.package_root changed from {base_root!r} (base) to {head_root!r} '
            f'(head). The scan surface cannot be narrowed by the PR it gates.'
        )

    def _merged(raw: dict[str, Any], layout: dict[str, Any]) -> set[str]:
        """The excludes one revision actually scans with: repo-wide plus per-gate."""
        entries = list(layout.get('excludes', []) or [])
        if gate is not None:
            gates = raw.get('gates', {})
            body = gates.get(gate, {}) if isinstance(gates, dict) else {}
            if isinstance(body, dict):
                entries += list(body.get('excludes', []) or [])
        return {entry for entry in entries if isinstance(entry, str)}

    added = _merged(config(banner), head_layout) - _merged(base_raw, base_layout)
    if added:
        where = 'layout.excludes' if gate is None else f'layout/gates.{gate} excludes'
        failures.append(
            f'{where} added in head that are not in base: {sorted(added)!r}. '
            f'New excludes hide files from the ratchet; add them in a separate PR '
            f'that ratchets the totals first.'
        )
    return failures


def significant_lines(path: Path) -> int:
    """Count non-blank, non-comment-only lines — the unit budgets and the
    test/code ratio are measured in."""
    count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            count += 1
    return count


def _is_path_excluded(rel: Path, excludes: list[str]) -> bool:
    # Path-part match, not substring: an exclude entry matches only if its
    # parts appear as a contiguous slice of rel's parts, so 'dist' does not
    # spuriously match 'talos/distance.py'.
    parts = rel.parts
    for ex in excludes:
        ex_parts = Path(ex).parts
        if not ex_parts:
            continue
        width = len(ex_parts)
        for i in range(max(0, len(parts) - width + 1)):
            if parts[i:i + width] == ex_parts:
                return True
    return False


def find_python_files(root: Path, excludes: list[str]) -> list[Path]:
    """Every `*.py` under root whose path is not excluded, sorted."""
    return [
        path for path in sorted([root] if root.is_file() else root.rglob('*.py'))
        if not _is_path_excluded(path.relative_to(REPO_ROOT), excludes)
    ]


# Re-exported so call sites can catch a parse failure without importing the
# TOML module themselves, which is what let the unguarded import spread.
TOMLDecodeError: Final[type[Exception]] = tomllib.TOMLDecodeError


def loads_toml(text: str) -> dict[str, Any]:
    """Parse TOML text, tolerating a pre-3.11 interpreter.

    The single place this repository parses TOML. `tomllib` arrived in 3.11, so
    a repository derived from this template with a lower floor resolves the
    `tomli` fallback instead. Every gate and contract test goes through here,
    so the fallback cannot be forgotten at one call site -- and no `try`/
    `except` lands under `tests/`, where the test-fallback gate would read it
    as a swallowed assertion.
    """
    return tomllib.loads(text)


def comparison_ref(base_ref: str, head_ref: str = 'HEAD') -> str:
    """Use the evidenced pre-adoption boundary once; governed bases always win."""
    import subprocess

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(['git', *args], cwd=REPO_ROOT, text=True,
                              capture_output=True, check=False)

    if run('rev-parse', '--verify', f'{base_ref}^{{commit}}').returncode:
        fail_setup('ADOPTION BOUNDARY', f'base ref is unreachable: {base_ref}')
    if run('cat-file', '-e', f'{base_ref}:governance.yml').returncode == 0:
        return base_ref
    anchor = section_setting('adoption', 'baseline_commit', '', 'ADOPTION BOUNDARY')
    if not anchor:
        return base_ref
    if not re.fullmatch(r'[0-9a-f]{40}', anchor):
        fail_setup('ADOPTION BOUNDARY', 'adoption.baseline_commit must be a full commit SHA')
    if run('merge-base', '--is-ancestor', base_ref, anchor).returncode:
        fail_setup('ADOPTION BOUNDARY', 'protected base is not an ancestor of the adoption boundary')
    if run('merge-base', '--is-ancestor', anchor, head_ref).returncode:
        fail_setup('ADOPTION BOUNDARY', 'adoption boundary is not an ancestor of the judged head')
    if run('cat-file', '-e', f'{anchor}:governance.yml').returncode == 0:
        fail_setup('ADOPTION BOUNDARY', 'adoption boundary already carries governance')
    return anchor
