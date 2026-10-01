#!/usr/bin/env python3
"""Typing gate — mechanical enforcement of type discipline.

This gate blocks a PR that:

  1. Weakens the pyright configuration (gate-config ratchet).
  2. Adds a ``pyrightconfig.json`` anywhere in the repo (pyright would
     prefer it over ``pyproject.toml`` and bypass the strict config).
  3. Changes ``pyright.include`` away from the required package list.
  4. Introduces new ``Any`` in production code (escape-hatch ratchet).
  5. Introduces new ``# type: ignore`` / ``# pyright: ignore`` / ``# noqa``
     comments (escape-hatch ratchet).
  6. Introduces new ``cast(..., Any)`` / ``cast(Any, ...)`` calls.
  7. Increases the total pyright-strict error count (pyright-error ratchet).
  8. Raises ANY budget value compared to the protected base ref
     (budget-source ratchet — the oracle cannot be weakened by the
     same PR it gates).
  9. Deletes a pattern key from the budget (same bypass class as #8).
  10. Reports ``filesAnalyzed`` below the number of Python files under the
      package root (shrinking the analysis surface is a trivial bypass).

The gate is a ratchet, not a flat hard-fail. The budget file at
``.github/budgets.json`` ``typing`` caps the total count of each escape-hatch
pattern and the pyright error count. Exceeding any cap fails the build.
Decreasing the cap is allowed — a PR may lower the numbers to lock in
improvements — but all decreases are checked against the base-ref
budget so the oracle cannot be weakened in the same PR that gates.

Usage:

  python governance/typing_gate.py                                # head-only checks
  python governance/typing_gate.py --pyright-json <path>          # + pyright ratchet
  python governance/typing_gate.py --base-budget <path>           # + base-vs-head ratchet
  python governance/typing_gate.py --update-budget [--pyright-json <path>]

Exit codes:

  0 — all gates pass
  1 — one or more gates failed
  2 — gate itself could not run (missing config, file-system error, etc.)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Final

from _common import (
    REPO_ROOT,
    TOMLDecodeError,
    exit_if_disabled,
    find_python_files,
    layout_excludes,
    loads_toml,
    resolve_package_dir,
    scan_surface_failures,
    write_budget_section,
)

BANNER: Final[str] = 'TYPING GATE'
BUDGET_PATH: Final[Path] = REPO_ROOT / '.github' / 'budgets.json'
BUDGET_SECTION: Final[str] = 'typing'
PYPROJECT_PATH: Final[Path] = REPO_ROOT / 'pyproject.toml'


def _setup_failure(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


# -------------------------------------------------------------------
# File walking
# -------------------------------------------------------------------

def count_pattern(files: list[Path], pattern: str) -> int:
    """Count non-overlapping matches of `pattern` across `files`. Any
    read or decode error is a fatal setup failure (not a silent skip);
    a silently skipped file would under-count escape hatches and let
    regressions through."""
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        raise SystemExit(
            f'typing_gate: invalid regex in budget ({pattern!r}): {exc}'
        ) from exc
    total = 0
    for f in files:
        try:
            text = f.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(
                f'typing_gate: cannot read {f}: {exc}'
            ) from exc
        total += len(rx.findall(text))
    return total


# -------------------------------------------------------------------
# AST-based Any detection.
#
# The regex ratchet catches bare ``Any`` (``x: Any``) but is blind to
# qualified / aliased forms: ``typing.Any``, ``t.Any`` after ``import
# typing as t``, ``A`` after ``from typing import Any as A``, and any
# chained attribute access like ``typing.Any`` inside a ``cast()``.
# The AST walks the module, resolves which local names actually bind
# to ``typing.Any``, and counts every reference regardless of surface
# form.
# -------------------------------------------------------------------

def _collect_any_bindings(tree: ast.AST) -> tuple[set[str], set[str]]:
    """For one module AST, return:

      * ``direct`` — names that refer to ``typing.Any``. Populated by
        ``from typing import Any`` (key ``Any``), aliased forms like
        ``from typing import Any as X`` (key ``X``), AND by module-level
        assignment aliases such as ``A = typing.Any`` (key ``A``) or
        chained aliases ``B = A`` (key ``B``). Iteratively expanded to
        a fixed point so ``A = typing.Any; B = A; C = B`` all resolve.
      * ``module_aliases`` — names that refer to the ``typing`` module.
        Populated by ``import typing`` (``typing``) and ``import typing
        as t`` (``t``).
    """
    direct: set[str] = set()
    module_aliases: set[str] = set()

    # Pass 1: import statements.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'typing':
            for alias in node.names:
                if alias.name == 'Any':
                    direct.add(alias.asname or 'Any')
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'typing':
                    module_aliases.add(alias.asname or 'typing')

    # Pass 2: assignment aliases. Iterate to a fixed point so chained
    # aliases (``A = typing.Any``; ``B = A``; ``C = B``) all resolve.
    # Bounded at 16 rounds to prevent runaway; any real module converges
    # in a couple of passes.
    for _ in range(16):
        before = len(direct)
        for node in ast.walk(tree):
            value: ast.AST | None = None
            target_names: list[str] = []
            if isinstance(node, ast.Assign):
                value = node.value
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        target_names.append(t.id)
            elif isinstance(node, ast.AnnAssign):
                value = node.value
                if isinstance(node.target, ast.Name):
                    target_names.append(node.target.id)
            else:
                continue
            if value is None or not target_names:
                continue
            if _resolves_to_any(value, direct, module_aliases):
                for name in target_names:
                    direct.add(name)
        if len(direct) == before:
            break

    return direct, module_aliases


def _resolves_to_any(
    node: ast.AST,
    direct: set[str],
    module_aliases: set[str],
) -> bool:
    """True if evaluating ``node`` would produce ``typing.Any`` under
    the current direct / module-alias bindings. Used by
    ``_collect_any_bindings`` to expand assignment-alias chains.
    """
    if isinstance(node, ast.Name):
        return node.id in direct
    if isinstance(node, ast.Attribute) and node.attr == 'Any':
        target = node.value
        return isinstance(target, ast.Name) and target.id in module_aliases
    return False


def _is_any_reference(
    node: ast.AST,
    direct: set[str],
    module_aliases: set[str],
) -> bool:
    """True if ``node`` is a usage-site reference to ``typing.Any``.

    Catches bare ``Any`` / aliased-direct names and attribute access
    like ``typing.Any`` / ``t.Any``. Limits ``ast.Name`` matches to
    load context so we do not double-count the LHS of an assignment
    that also has a typing.Any reference on its RHS.
    """
    if isinstance(node, ast.Name):
        # Only Load context is a usage site. Store (LHS of =), Del, and
        # annotation targets (also Store) are bindings, not uses.
        if not isinstance(getattr(node, 'ctx', None), ast.Load):
            return False
        return node.id in direct
    if isinstance(node, ast.Attribute) and node.attr == 'Any':
        target = node.value
        return isinstance(target, ast.Name) and target.id in module_aliases
    return False


def count_any_references_ast(files: list[Path]) -> int:
    """Count every reference to ``typing.Any`` across ``files``,
    resolved through the module's own import / alias graph.

    Includes: annotations (``x: Any``, ``-> Any``), generic-arg positions
    (``list[Any]``, ``dict[str, Any]``), ``cast(Any, x)``, attribute
    forms (``typing.Any``, ``t.Any``), aliased imports (``from typing
    import Any as A`` → any use of ``A``), and every other site where
    the resolved name is ``typing.Any``.

    Excludes: strings containing the literal word ``Any``, comments,
    and the ``alias.name`` field of the ``from typing import Any``
    statement itself (not an ``ast.Name`` node).
    """
    total = 0
    for f in files:
        try:
            text = f.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(f'typing_gate: cannot read {f}: {exc}') from exc
        try:
            tree = ast.parse(text, filename=str(f))
        except SyntaxError as exc:
            raise SystemExit(
                f'typing_gate: cannot parse {f}: {exc}'
            ) from exc

        direct, module_aliases = _collect_any_bindings(tree)
        if not direct and not module_aliases:
            continue  # module has no way to refer to Any

        for node in ast.walk(tree):
            if _is_any_reference(node, direct, module_aliases):
                total += 1

    return total


# -------------------------------------------------------------------
# GATE 1 — pyright config must be strict and must ban explicit Any
# -------------------------------------------------------------------

# Required [tool.pyright] keys.
#
# The gate asserts these keys are PRESENT in pyproject.toml with the
# exact values below. That is a configuration-presence check: the
# config file carries the values.
#
# What actually enforces what at runtime is split:
#
#   * reportMissingImports, reportMissingTypeStubs, reportUnknown*,
#     reportMissingParameterType, reportConstantRedefinition,
#     reportImportCycles: ENFORCED by pyright itself. Every error
#     pyright emits under these rules is counted toward
#     summary.errorCount, which gate_pyright_errors() ratchets.
#
#   * reportExplicitAny: pyright 1.1.408 (the pinned version) emits
#     `Config contains unrecognized setting "reportExplicitAny"` and
#     does not honor it. We keep it required here anyway as a
#     forward-compatible config-presence check: the day pyright or its
#     successor respects the setting, enforcement kicks in without a
#     gate change. Today, the actual Any enforcement is the AST-based
#     gate_any_references_ast() below, which resolves every surface
#     form including module-level assignment aliases.
#
#   * typeCheckingMode: this must be "strict" so that all of the
#     strict-default report* rules above are active.
REQUIRED_PYRIGHT: Final[dict[str, object]] = {
    'typeCheckingMode': 'strict',
    'reportExplicitAny': 'error',
    'reportMissingImports': 'error',
    'reportMissingTypeStubs': 'error',
    'reportUnknownArgumentType': 'error',
    'reportUnknownMemberType': 'error',
    'reportUnknownVariableType': 'error',
    'reportUnknownLambdaType': 'error',
    'reportUnknownParameterType': 'error',
    'reportMissingParameterType': 'error',
    'reportConstantRedefinition': 'error',
    'reportImportCycles': 'error',
}

# The `include` list pyright must analyze. Shrinking this to an empty
# list or a non-existent path drops filesAnalyzed to zero, trivially
# passing the error-count ratchet. Gate asserts exact match.
REQUIRED_PYRIGHT_INCLUDE: Final[list[str]] = ['talos']

FORBIDDEN_VALUES: Final[frozenset[object]] = frozenset(
    {'none', 'warning', 'information', 'info', 'false', False}
)


_PRUNE_DIRS: Final[frozenset[str]] = frozenset(
    {'.git', '.venv', 'venv', 'node_modules', 'build', 'dist', '__pycache__'}
)


def _find_pyrightconfig_json() -> list[Path]:
    """Walk the repo for pyrightconfig.json files with directory
    pruning. ``Path.rglob`` descends into ``.git`` — which is huge on
    a fetch-depth=0 CI checkout — before filtering, so we use os.walk
    and remove prune targets from ``dirnames`` before recursing.
    """
    import os
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
        if 'pyrightconfig.json' in filenames:
            found.append(Path(dirpath) / 'pyrightconfig.json')
    return sorted(found)


def gate_pyright_config(config: dict[str, object]) -> list[str]:
    failures: list[str] = []

    # Pyright prefers pyrightconfig.json over [tool.pyright] in pyproject.toml.
    # A PR that drops such a file anywhere in the repo silently shadows every
    # setting this gate audits. Ban it outright.
    for cfg in _find_pyrightconfig_json():
        rel = cfg.relative_to(REPO_ROOT)
        failures.append(
            f'pyrightconfig.json found at {rel}. '
            f'Pyright prefers this file over pyproject.toml and would '
            f'bypass the strict [tool.pyright] config. Delete it.'
        )

    tool = config.get('tool')
    pyright = tool.get('pyright') if isinstance(tool, dict) else None
    if not isinstance(pyright, dict):
        failures.append('[tool.pyright] section is missing from pyproject.toml')
        return failures

    for key, required in REQUIRED_PYRIGHT.items():
        actual = pyright.get(key)
        if actual != required:
            failures.append(
                f'pyright.{key} must be {required!r}, got {actual!r}'
            )

    # `include` must match exactly. Shrinking it drops filesAnalyzed.
    actual_include = pyright.get('include')
    if actual_include != REQUIRED_PYRIGHT_INCLUDE:
        failures.append(
            f'pyright.include must be exactly {REQUIRED_PYRIGHT_INCLUDE!r}, '
            f'got {actual_include!r}. Changing this shrinks the analysis '
            f'surface and lets errors escape the gate.'
        )

    for key, value in pyright.items():
        if not isinstance(key, str):
            continue
        if key.startswith('report') and value in FORBIDDEN_VALUES:
            failures.append(
                f'pyright.{key} = {value!r} -- gate weakening disallowed; '
                f"must be 'error' (or absent to inherit strict default)"
            )

    return failures


# -------------------------------------------------------------------
# GATE 2 — escape-hatch pattern ratchet against committed budget
# -------------------------------------------------------------------

def gate_escape_hatch_ratchet(budget: dict[str, object]) -> list[str]:
    failures: list[str] = []
    package_root = resolve_package_dir(BANNER)

    excludes = layout_excludes('typing', BANNER)
    files = find_python_files(package_root, excludes)

    patterns = budget.get('patterns')
    if not isinstance(patterns, dict):
        return ['budgets.json "typing" must define patterns']

    for name, spec in patterns.items():
        if not isinstance(spec, dict):
            return [f'budgets.json "typing": pattern {name!r} must be an object']
        pattern = spec.get('pattern')
        if not isinstance(pattern, str) or not pattern:
            return [f'budgets.json "typing": pattern {name!r} has no regex']
        raw_total = spec.get('total', 0)
        if isinstance(raw_total, bool) or not isinstance(raw_total, int):
            return [
                f'budgets.json "typing": pattern {name!r} total must be a '
                f'non-negative integer (got {raw_total!r})'
            ]
        if raw_total < 0:
            return [
                f'budgets.json "typing": pattern {name!r} total must be '
                f'non-negative (got {raw_total})'
            ]
        current = count_pattern(files, pattern)
        if current > raw_total:
            failures.append(
                f'[{name}] budget={raw_total} current={current} '
                f'(pattern={pattern!r}) -- ratchet exceeded. '
                f'Remove the new escape hatch or lower the budget.'
            )

    return failures


# -------------------------------------------------------------------
# GATE 2b — AST-based Any-reference ratchet.
# Catches every surface form of typing.Any, not just bare `Any`.
# -------------------------------------------------------------------

def gate_any_references_ast(budget: dict[str, object]) -> list[str]:
    package_root = resolve_package_dir(BANNER)

    excludes = layout_excludes('typing', BANNER)
    files = find_python_files(package_root, excludes)

    spec = budget.get('any_references')
    if not isinstance(spec, dict):
        return [
            'budgets.json "typing" must include an any_references section '
            "with {'total': <int>}"
        ]
    raw_total = spec.get('total', 0)
    if isinstance(raw_total, bool) or not isinstance(raw_total, int):
        return [
            f'budgets.json "typing": any_references.total must be a '
            f'non-negative integer (got {raw_total!r})'
        ]
    if raw_total < 0:
        return [
            f'budgets.json "typing": any_references.total must be '
            f'non-negative (got {raw_total})'
        ]

    current = count_any_references_ast(files)
    if current > raw_total:
        return [
            f'any_references (AST): budget={raw_total} current={current} '
            f'(delta={current - raw_total}). Covers Any, typing.Any, '
            f't.Any, aliased imports. Remove the new Any or lower '
            f'the budget.'
        ]
    return []


# -------------------------------------------------------------------
# GATE 3 — pyright total error count ratchet
# -------------------------------------------------------------------

def gate_pyright_errors(
    pyright_json_path: str | None,
    budget: dict[str, object],
) -> list[str]:
    """Ratchet both pyright severities against their budgets.

    Errors alone leave a real class of defect ungated: under strict mode
    pyright reports plenty as warnings, and a repository could accumulate
    them indefinitely while the gate stayed green.
    """
    failures: list[str] = []
    for severity, summary_key, section in (
        ('error', 'errorCount', 'pyright_errors'),
        ('warning', 'warningCount', 'pyright_warnings'),
    ):
        failures.extend(
            _gate_pyright_severity(pyright_json_path, budget, severity, summary_key, section)
        )
    return failures


def _gate_pyright_severity(
    pyright_json_path: str | None,
    budget: dict[str, object],
    severity: str,
    summary_key: str,
    section: str,
) -> list[str]:
    if pyright_json_path is None:
        return []
    path = Path(pyright_json_path)
    if not path.is_file():
        return [f'pyright output not found at {path}']
    try:
        raw_text = path.read_text(encoding='utf-8')
    except OSError as exc:
        return [f'pyright output at {path} could not be read: {exc}']
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return [f'pyright output is not valid JSON: {e}']

    if not isinstance(data, dict):
        return ['pyright output must be a JSON object']
    summary = data.get('summary', {})
    if not isinstance(summary, dict):
        return ['pyright output .summary must be an object']
    raw_current = summary.get(summary_key, 0)
    if isinstance(raw_current, bool) or not isinstance(raw_current, int):
        return [
            f'pyright output .summary.{summary_key} must be an integer '
            f'(got {raw_current!r})'
        ]
    current_errors = raw_current

    py_budget_raw = budget.get(section)
    if not isinstance(py_budget_raw, dict):
        return [
            f'budgets.json "typing" must include a {section} section '
            "with {'total': <int>}"
        ]
    raw_budget_total = py_budget_raw.get('total', 0)
    if isinstance(raw_budget_total, bool) or not isinstance(raw_budget_total, int):
        return [
            f'budgets.json "typing": {section}.total must be a '
            f'non-negative integer (got {raw_budget_total!r})'
        ]
    if raw_budget_total < 0:
        return [
            f'budgets.json "typing": {section}.total must be '
            f'non-negative (got {raw_budget_total})'
        ]
    budget_total = raw_budget_total

    if current_errors > budget_total:
        # Per-rule summary for the failure message.
        per_rule: dict[str, int] = {}
        diagnostics = data.get('generalDiagnostics') if isinstance(data, dict) else None
        if isinstance(diagnostics, list):
            for diag in diagnostics:
                if not isinstance(diag, dict):
                    continue
                if diag.get('severity') != severity:
                    continue
                rule = str(diag.get('rule', '<no-rule>'))
                per_rule[rule] = per_rule.get(rule, 0) + 1
        top = sorted(per_rule.items(), key=lambda kv: -kv[1])[:5]
        top_str = '; '.join(f'{r}={n}' for r, n in top)
        return [
            f'pyright {severity}s: budget={budget_total} current={current_errors} '
            f'(delta={current_errors - budget_total}). Top rules: {top_str}'
        ]

    return []


# -------------------------------------------------------------------
# GATE 4 — budget-source ratchet. The PR's own budget (``head``) must
# not raise any value and must not delete any pattern, compared to
# the budget at the protected base ref. Without this, a PR can raise
# its own ceiling and pass.
# -------------------------------------------------------------------

def gate_budget_source(
    base_budget_path: str | None,
    base_config_path: str | None,
    bootstrap: bool,
    head_budget: dict[str, object],
) -> list[str]:
    """Compare the head budget to the budget at the protected base ref.
    The PR cannot raise its own ceiling. Two legal states:

      * ``--base-budget PATH`` with PATH existing and valid JSON: do the
        base-vs-head comparison.
      * ``--bootstrap``: explicit first-commit override for the PR that
        introduces ``.github/budgets.json`` to main. The workflow
        selects this mode mechanically by diffing against the base ref.

    Any other state -- including a missing ``--base-budget`` file
    without ``--bootstrap`` -- is a hard failure. A graceful skip on
    missing file would convert ``someone deleted the budget from main``
    into ``nothing to enforce'', which is the pattern this gate exists
    to prevent.
    """

    if bootstrap:
        if base_budget_path is not None:
            return [
                'typing_gate: --bootstrap and --base-budget are mutually '
                'exclusive; pass exactly one.'
            ]
        # Bootstrap mode: the PR introduces the budget to main. The
        # workflow confirmed this by checking that the head commit adds
        # .github/budgets.json. No base-vs-head comparison in this
        # case -- there is no base.
        return []

    if base_budget_path is None:
        return [
            'typing_gate: neither --base-budget nor --bootstrap was given. '
            'The workflow must either (a) fetch the budget from the '
            'protected base ref and pass --base-budget PATH, or (b) pass '
            '--bootstrap if this commit introduces the budget.'
        ]

    base_path = Path(base_budget_path)
    if not base_path.is_file():
        return [
            f'typing_gate: base-ref budget not found at {base_path} and '
            f'--bootstrap was not given. Either (a) the budget was '
            f'deleted from the protected base ref -- restore it -- or '
            f'(b) this is the commit introducing the budget, in which '
            f'case pass --bootstrap. Silent skip is not an option.'
        ]

    try:
        base_budget = json.loads(base_path.read_text(encoding='utf-8')).get(BUDGET_SECTION, {})
    except (OSError, json.JSONDecodeError) as exc:
        return [f'typing_gate: cannot read base budget {base_path}: {exc}']

    if not isinstance(base_budget, dict):
        return [f'typing_gate: base budget {base_path} is not a JSON object']

    failures: list[str] = []

    # Structural invariants: the scan surface cannot be narrowed.
    #
    # * package_root identical: cannot point the gate at a smaller subtree.
    # * excludes: head must be a subset of base (can remove, never add).
    #   Adding an exclude hides files from the ratchet.
    failures.extend(scan_surface_failures(base_config_path, BANNER, 'typing'))

    # Per-pattern regex identity: a regex in head for a key present in
    # base must be the exact same regex. Otherwise a PR could rewrite
    # the regex to one that never matches and keep the total at zero,
    # neutering that ratchet key.
    base_patterns_raw = base_budget.get('patterns')
    head_patterns_raw = head_budget.get('patterns')
    base_patterns: dict[str, object] = (
        base_patterns_raw if isinstance(base_patterns_raw, dict) else {}
    )
    head_patterns: dict[str, object] = (
        head_patterns_raw if isinstance(head_patterns_raw, dict) else {}
    )
    for key, base_spec in base_patterns.items():
        if key not in head_patterns:
            continue  # handled below by key-preservation check
        head_spec = head_patterns[key]
        if not (isinstance(base_spec, dict) and isinstance(head_spec, dict)):
            continue
        base_rx = base_spec.get('pattern')
        head_rx = head_spec.get('pattern')
        if base_rx != head_rx:
            failures.append(
                f'pattern {key!r} regex changed from {base_rx!r} (base) '
                f'to {head_rx!r} (head). Regex strings for keys present '
                f'in the base budget must remain identical; rewriting '
                f'the regex can nullify the ratchet while leaving the '
                f'total unchanged.'
            )

    # Top-level integer-ceiling sections (pyright_errors, any_references).
    for section in ('pyright_errors', 'pyright_warnings', 'any_references'):
        base_sec = base_budget.get(section)
        head_sec = head_budget.get(section)
        if not isinstance(base_sec, dict):
            continue
        if not isinstance(head_sec, dict):
            failures.append(
                f'{section} section was deleted from the head budget. '
                f'Keys present in the base-ref budget must be preserved.'
            )
            continue
        b_total = base_sec.get('total', 0)
        h_total = head_sec.get('total', 0)
        if (
            isinstance(b_total, int)
            and not isinstance(b_total, bool)
            and isinstance(h_total, int)
            and not isinstance(h_total, bool)
            and h_total > b_total
        ):
            failures.append(
                f'{section}.total was raised from {b_total} (base) '
                f'to {h_total} (head). The oracle cannot be weakened '
                f'by the PR it gates.'
            )

    # Pattern key preservation and per-key total ratchet. base_patterns
    # and head_patterns are already extracted above for the regex-
    # identity check; reuse.
    for key in base_patterns:
        if key not in head_patterns:
            failures.append(
                f'pattern {key!r} was deleted from the budget. '
                f'Keys present in the base-ref budget must be preserved.'
            )

    for key, base_spec in base_patterns.items():
        if key not in head_patterns:
            continue
        head_spec = head_patterns[key]
        if not (isinstance(base_spec, dict) and isinstance(head_spec, dict)):
            continue
        b_total = base_spec.get('total', 0)
        h_total = head_spec.get('total', 0)
        if (
            isinstance(b_total, int)
            and not isinstance(b_total, bool)
            and isinstance(h_total, int)
            and not isinstance(h_total, bool)
            and h_total > b_total
        ):
            failures.append(
                f'pattern {key!r} total was raised from {b_total} (base) '
                f'to {h_total} (head). Only decreases are allowed.'
            )

    return failures


# -------------------------------------------------------------------
# GATE 5 — pyright filesAnalyzed must match the number of Python
# files under the package root. Shrinking include (or hiding files
# behind exclude) trivially drops filesAnalyzed to a smaller set and
# lets errors escape.
# -------------------------------------------------------------------

def gate_files_analyzed(
    pyright_json_path: str | None,
    budget: dict[str, object],
) -> list[str]:
    if pyright_json_path is None:
        return []
    path = Path(pyright_json_path)
    if not path.is_file():
        return []

    try:
        raw_text = path.read_text(encoding='utf-8')
    except OSError as exc:
        return [f'pyright output at {path} could not be read: {exc}']
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return [f'pyright output is not valid JSON: {e}']
    if not isinstance(data, dict):
        return ['pyright output must be a JSON object']
    summary = data.get('summary', {})
    if not isinstance(summary, dict):
        return ['pyright output .summary must be an object']

    raw_analyzed = summary.get('filesAnalyzed', 0)
    if isinstance(raw_analyzed, bool) or not isinstance(raw_analyzed, int):
        return [
            f'pyright output .summary.filesAnalyzed must be an integer '
            f'(got {raw_analyzed!r})'
        ]
    analyzed = raw_analyzed

    package_root = resolve_package_dir(BANNER)
    excludes = layout_excludes('typing', BANNER)
    expected = len(find_python_files(package_root, excludes))

    if analyzed < expected:
        return [
            f'pyright filesAnalyzed={analyzed} but the package has '
            f'{expected} Python files. The analysis surface was '
            f'shrunk (likely via pyright.include or pyright.exclude) '
            f'and errors are hidden outside the analyzed set.'
        ]
    return []


# -------------------------------------------------------------------
# Baseline regeneration (--update-budget)
# -------------------------------------------------------------------

DEFAULT_PATTERNS: Final[dict[str, dict[str, object]]] = {
    'any_annotation':  {'pattern': r':\s*Any\b', 'total': 0},
    'any_return':      {'pattern': r'->\s*Any\b', 'total': 0},
    'any_import':      {'pattern': r'from typing import[^\n]*\bAny\b', 'total': 0},
    'cast_any':        {'pattern': r'cast\([^)]*\bAny\b', 'total': 0},
    'dict_any':        {'pattern': r'dict\[[^]]*\bAny\b', 'total': 0},
    'list_any':        {'pattern': r'list\[\s*Any\b', 'total': 0},
    'tuple_any':       {'pattern': r'tuple\[[^]]*\bAny\b', 'total': 0},
    'type_ignore':     {'pattern': r'#\s*type:\s*ignore', 'total': 0},
    'pyright_ignore':  {'pattern': r'#\s*pyright:\s*ignore', 'total': 0},
    'noqa':            {'pattern': r'#\s*noqa', 'total': 0},
}


def update_budget(pyright_json_path: str | None) -> None:
    if BUDGET_PATH.exists():
        budget = json.loads(BUDGET_PATH.read_text(encoding='utf-8')).get(BUDGET_SECTION, {})
    else:
        budget = {
            'schema_version': 2,
            'package_root': 'talos',
            'excludes': ['__pycache__', 'build', 'dist'],
            'patterns': {k: dict(v) for k, v in DEFAULT_PATTERNS.items()},
            'any_references': {'total': 0},
            'pyright_errors': {'total': 0},
            'pyright_warnings': {'total': 0},
        }

    # Migrate schema v1 -> v2 (add any_references if missing)
    if 'any_references' not in budget:
        budget['any_references'] = {'total': 0}
        budget['schema_version'] = 2

    package_root = resolve_package_dir(BANNER)
    excludes = layout_excludes('typing', BANNER)
    files = find_python_files(package_root, excludes)

    for _name, spec in budget['patterns'].items():
        spec['total'] = count_pattern(files, str(spec['pattern']))

    budget['any_references']['total'] = count_any_references_ast(files)

    if pyright_json_path is not None:
        try:
            data = json.loads(Path(pyright_json_path).read_text(encoding='utf-8'))
            summary = data.get('summary', {})
            budget['pyright_errors']['total'] = int(summary.get('errorCount', 0))
            budget.setdefault('pyright_warnings', {'total': 0})
            budget['pyright_warnings']['total'] = int(summary.get('warningCount', 0))
        except (OSError, json.JSONDecodeError) as e:
            print(f'warning: could not read pyright output: {e}', file=sys.stderr)

    BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_budget_section(BUDGET_SECTION, budget, BANNER)

    total_hatches = sum(int(s['total']) for s in budget['patterns'].values())
    any_refs = int(budget['any_references']['total'])
    pyright_total = int(budget.get('pyright_errors', {}).get('total', 0))
    print(f'budget updated -> {BUDGET_PATH.relative_to(REPO_ROOT)}')
    print(f'  regex escape-hatch occurrences: {total_hatches}')
    print(f'  AST Any references:             {any_refs}')
    print(f'  pyright errors:                 {pyright_total}')


# -------------------------------------------------------------------
# Orchestration
# -------------------------------------------------------------------

def load_pyproject() -> dict[str, object]:
    try:
        return loads_toml(PYPROJECT_PATH.read_text(encoding='utf-8'))
    except OSError as exc:
        _setup_failure(
            f'typing_gate: cannot read {PYPROJECT_PATH.relative_to(REPO_ROOT)}: {exc}'
        )
    except TOMLDecodeError as exc:
        _setup_failure(
            f'typing_gate: cannot parse {PYPROJECT_PATH.relative_to(REPO_ROOT)}: {exc}'
        )


def load_budget() -> dict[str, object]:
    if not BUDGET_PATH.exists():
        _setup_failure(
            f'typing gate cannot run: {BUDGET_PATH.relative_to(REPO_ROOT)} '
            f'is missing. Run with --update-budget to create it.'
        )
    try:
        return json.loads(BUDGET_PATH.read_text(encoding='utf-8')).get(BUDGET_SECTION, {})
    except OSError as exc:
        _setup_failure(
            f'typing_gate: cannot read {BUDGET_PATH.relative_to(REPO_ROOT)}: {exc}'
        )
    except json.JSONDecodeError as exc:
        _setup_failure(
            f'typing_gate: cannot parse {BUDGET_PATH.relative_to(REPO_ROOT)}: {exc}'
        )


def main() -> int:
    exit_if_disabled('typing', BANNER)
    parser = argparse.ArgumentParser(description='Typing gate')
    parser.add_argument(
        '--pyright-json',
        default=None,
        help=(
            'Path to pyright --outputjson output. REQUIRED for normal gate '
            'runs (omitted only when --update-budget is set and pyright '
            'output is not available). Without it the pyright-error and '
            'filesAnalyzed ratchets cannot run; the gate therefore '
            'hard-fails if this flag is absent on a normal run.'
        ),
    )
    parser.add_argument(
        '--base-config',
        help='path to governance.yml at the protected base ref',
    )
    parser.add_argument(
        '--base-budget',
        default=None,
        help=(
            'Path to the budget JSON at the protected base ref. Without '
            'this, the gate cannot check whether the head PR has raised '
            'its own ceiling.'
        ),
    )
    parser.add_argument(
        '--bootstrap',
        action='store_true',
        help=(
            'Explicit first-commit override: skip the base-vs-head '
            'comparison because this commit introduces the budget to '
            'the protected base ref. The workflow selects this mode '
            'mechanically by diffing against the base ref; it must not '
            'be set by hand.'
        ),
    )
    parser.add_argument(
        '--update-budget',
        action='store_true',
        help='Regenerate .github/budgets.json from current repo state',
    )
    args = parser.parse_args()

    if args.update_budget:
        update_budget(args.pyright_json)
        return 0

    # Normal gate runs must supply the pyright output. A workflow edit
    # that drops the flag would otherwise silently skip the pyright-
    # error and filesAnalyzed ratchets. We refuse to run instead of
    # producing a green gate on half the checks.
    if args.pyright_json is None:
        print(
            'TYPING GATE -- FAIL\n\n'
            '  gate: runtime-args\n'
            '    - --pyright-json was not provided. Normal gate runs require '
            'pyright output so the pyright-error-ratchet and files-analyzed-'
            'ratchet can run. Pass --pyright-json PATH or, if regenerating '
            'the budget, use --update-budget instead.\n\n'
            '1 failure(s). Merge blocked.',
            file=sys.stdout,
        )
        return 1

    config = load_pyproject()
    budget = load_budget()

    failures: list[tuple[str, str]] = []

    for msg in gate_pyright_config(config):
        failures.append(('pyright-config', msg))

    for msg in gate_budget_source(
        args.base_budget, args.base_config, args.bootstrap, budget
    ):
        failures.append(('budget-source-ratchet', msg))

    for msg in gate_escape_hatch_ratchet(budget):
        failures.append(('escape-hatch-ratchet', msg))

    for msg in gate_any_references_ast(budget):
        failures.append(('any-references-ast-ratchet', msg))

    for msg in gate_pyright_errors(args.pyright_json, budget):
        failures.append(('pyright-error-ratchet', msg))

    for msg in gate_files_analyzed(args.pyright_json, budget):
        failures.append(('files-analyzed-ratchet', msg))

    if failures:
        print('TYPING GATE -- FAIL')
        print('')
        by_gate: dict[str, list[str]] = {}
        for gate, msg in failures:
            by_gate.setdefault(gate, []).append(msg)
        for gate_name, msgs in by_gate.items():
            print(f'  gate: {gate_name}')
            for m in msgs:
                print(f'    - {m}')
        print('')
        print(f'{len(failures)} failure(s). Merge blocked.')
        return 1

    print('TYPING GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
