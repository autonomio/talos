#!/usr/bin/env python3
"""Fail-loud gate -- ratchet on silent-fallback patterns.

This gate counts seven AST-detectable silent-fallback categories in
the production package and refuses any PR that grows any category's
total. Same base-vs-head ratchet as the typing gate: the head budget
cannot be weakened in the same PR that runs against it.

Categories enforced:

  1. bare_except          -- `except:` with no exception type.
  2. empty_pass           -- exception handler whose body is `pass`.
  3. empty_ellipsis       -- handler body is `...` (Ellipsis literal).
  4. empty_return_none    -- handler body is `return` or `return None`.
  5. empty_continue_break -- handler body is a single `continue` or
                             `break` (i.e. silently skip this
                             iteration).
  6. contextlib_suppress  -- call to `contextlib.suppress(...)` or an
                             alias thereof; resolves `import contextlib
                             as X`, `from contextlib import suppress`,
                             `from contextlib import suppress as Y`.
  7. errors_ignore_kwarg  -- any function call with keyword argument
                             `errors='ignore'`. Covers `str.encode`,
                             `bytes.decode`, `Path.read_text`, etc.

Usage:

  python governance/fail_loud_gate.py \\
    --base-budget <path>         # .github/budgets.json at BASE
    [--update-budget]            # regenerate the committed budget
    [--bootstrap]                # first-commit override, mirrors typing_gate

Exit codes:

  0 -- all gates pass
  1 -- one or more gates failed
  2 -- gate itself could not run (bad args, parse failure, etc.)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Final

from _common import (
    REPO_ROOT,
    exit_if_disabled,
    find_python_files,
    layout_excludes,
    resolve_package_dir,
    scan_surface_failures,
    write_budget_section,
)

BUDGET_PATH: Final[Path] = REPO_ROOT / '.github' / 'budgets.json'
BUDGET_SECTION: Final[str] = 'fail_loud'
BANNER: Final[str] = 'FAIL LOUD GATE'

CATEGORIES: Final[tuple[str, ...]] = (
    'bare_except',
    'empty_pass',
    'empty_ellipsis',
    'empty_return_none',
    'empty_continue_break',
    'contextlib_suppress',
    'errors_ignore_kwarg',
)


# --------------------------------------------------------------------
# File discovery + violation counting
# --------------------------------------------------------------------

def _collect_contextlib_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Return (module_aliases, direct_names) where:
      module_aliases -- names that refer to the contextlib module,
                        populated by `import contextlib [as X]` AND by
                        module-level assignments whose RHS resolves to
                        the contextlib module (`mod = contextlib`,
                        `mod2 = mod`, ...);
      direct_names   -- names that refer to contextlib.suppress,
                        populated by `from contextlib import suppress
                        [as Y]` AND by module-level assignments whose
                        RHS resolves to contextlib.suppress
                        (`sup = contextlib.suppress`,
                        `sup = mod.suppress`, `sup2 = sup`, ...).

    Both sets are expanded to a fixed point so arbitrarily chained
    aliases of either kind are resolved. Bounded at 16 rounds; real
    modules converge in 1-2.
    """
    module_aliases: set[str] = set()
    direct_names: set[str] = set()

    if not isinstance(tree, ast.Module):
        return module_aliases, direct_names

    # Pass 1: module-level import statements.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'contextlib':
                    module_aliases.add(alias.asname or 'contextlib')
        elif isinstance(node, ast.ImportFrom) and node.module == 'contextlib':
            for alias in node.names:
                if alias.name == 'suppress':
                    direct_names.add(alias.asname or 'suppress')

    # Pass 2: fixed point over module-level assignments. Each round may
    # grow module_aliases (via `mod = contextlib`) and direct_names
    # (via `sup = contextlib.suppress` or `sup = mod.suppress`). Stops
    # when neither set changes in a round.
    for _ in range(16):
        before = (len(module_aliases), len(direct_names))
        for node in tree.body:
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
            # RHS refers to the contextlib module directly?
            if isinstance(value, ast.Name) and value.id in module_aliases:
                module_aliases.update(target_names)
                continue
            # RHS resolves to contextlib.suppress?
            if _resolves_to_suppress(value, direct_names, module_aliases):
                direct_names.update(target_names)
        if (len(module_aliases), len(direct_names)) == before:
            break

    return module_aliases, direct_names


def _resolves_to_suppress(
    node: ast.AST,
    direct_names: set[str],
    module_aliases: set[str],
) -> bool:
    """True if evaluating ``node`` would yield ``contextlib.suppress``
    under the current binding sets. Used to expand assignment-alias
    chains in ``_collect_contextlib_names``."""
    if isinstance(node, ast.Name):
        return node.id in direct_names
    if isinstance(node, ast.Attribute) and node.attr == 'suppress':
        target = node.value
        return isinstance(target, ast.Name) and target.id in module_aliases
    return False


def _count_in_tree(tree: ast.AST) -> dict[str, int]:
    out = dict.fromkeys(CATEGORIES, 0)
    module_aliases, direct_names = _collect_contextlib_names(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                # bare_except: `except:` with no type
                if h.type is None:
                    out['bare_except'] += 1
                # empty-handler categories only fire when body is a
                # single trivial statement
                if len(h.body) == 1:
                    stmt = h.body[0]
                    if isinstance(stmt, ast.Pass):
                        out['empty_pass'] += 1
                    elif (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and stmt.value.value is Ellipsis
                    ):
                        out['empty_ellipsis'] += 1
                    elif (
                        isinstance(stmt, ast.Return)
                        and (
                            stmt.value is None
                            or (
                                isinstance(stmt.value, ast.Constant)
                                and stmt.value.value is None
                            )
                        )
                    ):
                        out['empty_return_none'] += 1
                    elif isinstance(stmt, (ast.Continue, ast.Break)):
                        out['empty_continue_break'] += 1
        if isinstance(node, ast.Call):
            func = node.func
            is_suppress = False
            if isinstance(func, ast.Attribute) and func.attr == 'suppress':
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id in module_aliases
                ):
                    is_suppress = True
            elif isinstance(func, ast.Name) and func.id in direct_names:
                is_suppress = True
            if is_suppress:
                out['contextlib_suppress'] += 1
            for kw in node.keywords:
                if (
                    kw.arg == 'errors'
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == 'ignore'
                ):
                    out['errors_ignore_kwarg'] += 1
    return out


def count_violations(files: list[Path]) -> dict[str, int]:
    totals = dict.fromkeys(CATEGORIES, 0)
    for f in files:
        try:
            text = f.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            raise SystemExit(
                f'fail_loud_gate: cannot read {f}: {exc}'
            ) from exc
        try:
            tree = ast.parse(text, filename=str(f))
        except SyntaxError as exc:
            raise SystemExit(
                f'fail_loud_gate: cannot parse {f}: {exc}'
            ) from exc
        for k, v in _count_in_tree(tree).items():
            totals[k] += v
    return totals


# --------------------------------------------------------------------
# Gate logic
# --------------------------------------------------------------------

def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        raise SystemExit(f'fail_loud_gate: cannot read {label} {path}: {exc}') from exc
    try:
        data = json.loads(text).get(BUDGET_SECTION, {})
    except json.JSONDecodeError as exc:
        raise SystemExit(f'fail_loud_gate: {label} {path} is not valid JSON: {exc}') from exc
    if not isinstance(data, dict):
        raise SystemExit(f'fail_loud_gate: {label} {path} is not a JSON object')
    return data


def _get_int(d: dict[str, object], key: str, field_label: str) -> int:
    """Return `d[key]` as a non-negative int; raise SystemExit(2) with
    a clear message naming ``field_label`` if the value is missing or
    wrong-shaped. ``field_label`` is the full JSON-path form of the
    offending field (e.g. `categories.bare_except.total`)."""
    val = d.get(key, 0)
    if isinstance(val, bool) or not isinstance(val, int):
        print(
            f'fail_loud_gate: {field_label} must be a non-negative integer '
            f'(got {val!r})',
            file=sys.stderr,
        )
        raise SystemExit(2)
    if val < 0:
        print(
            f'fail_loud_gate: {field_label} must be non-negative (got {val})',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return val


def _category_total(budget: dict[str, object], cat: str) -> int:
    cats = budget.get('categories')
    if not isinstance(cats, dict):
        print('fail_loud_gate: budget missing `categories` mapping', file=sys.stderr)
        raise SystemExit(2)
    spec = cats.get(cat, {})
    if not isinstance(spec, dict):
        print(
            f'fail_loud_gate: budget.categories.{cat} is not an object',
            file=sys.stderr,
        )
        raise SystemExit(2)
    return _get_int(spec, 'total', f'categories.{cat}.total')


def gate(
    budget_head: dict[str, object],
    budget_base: dict[str, object] | None,
    base_config_path: str | None = None,
) -> list[str]:
    failures: list[str] = []

    # Structural: head must not narrow the scan surface, must preserve
    # every category key, and must not raise any total.
    if budget_base is not None:
        failures.extend(scan_surface_failures(base_config_path, BANNER, 'fail_loud'))
        base_cats = set((budget_base.get('categories') or {}).keys())
        head_cats = set((budget_head.get('categories') or {}).keys())
        for missing in base_cats - head_cats:
            failures.append(
                f'category {missing!r} deleted from head budget. Keys present '
                f'in the base-ref budget must be preserved.'
            )
        for cat in CATEGORIES:
            if cat in base_cats and cat in head_cats:
                base_total = _category_total(budget_base, cat)
                head_total = _category_total(budget_head, cat)
                if head_total > base_total:
                    failures.append(
                        f'categories.{cat}.total raised from {base_total} '
                        f'(base) to {head_total} (head). Only decreases allowed.'
                    )

    # Actual violation count vs head budget.
    package_root = resolve_package_dir(BANNER)
    excludes = layout_excludes('fail_loud', BANNER)
    files = find_python_files(package_root, excludes)
    current = count_violations(files)

    for cat in CATEGORIES:
        budget_total = _category_total(budget_head, cat)
        if current[cat] > budget_total:
            failures.append(
                f'{cat}: budget={budget_total} current={current[cat]} '
                f'(delta={current[cat] - budget_total}). Remove the new '
                f'silent fallback or lower the budget.'
            )

    return failures


# --------------------------------------------------------------------
# --update-budget helper
# --------------------------------------------------------------------

DEFAULT_BUDGET: Final[dict[str, object]] = {
    'schema_version': 1,
    'categories': {cat: {'total': 0} for cat in CATEGORIES},
}


def update_budget() -> None:
    if BUDGET_PATH.exists():
        budget = _load_json(BUDGET_PATH, 'budget')
    else:
        budget = json.loads(json.dumps(DEFAULT_BUDGET))

    package_root = resolve_package_dir(BANNER)
    excludes = layout_excludes('fail_loud', BANNER)
    files = find_python_files(package_root, excludes)
    current = count_violations(files)

    cats = budget.setdefault('categories', {})
    if not isinstance(cats, dict):
        raise SystemExit('fail_loud_gate: categories must be an object')
    for cat in CATEGORIES:
        cats.setdefault(cat, {})
        cats[cat]['total'] = current[cat]

    BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_budget_section(BUDGET_SECTION, budget, BANNER)
    print(f'budget updated -> {BUDGET_PATH.relative_to(REPO_ROOT)}')
    for cat in CATEGORIES:
        print(f'  {cat:<25} {current[cat]}')


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------

def main() -> int:
    exit_if_disabled('fail_loud', BANNER)
    parser = argparse.ArgumentParser(description='Fail-loud gate')
    parser.add_argument(
        '--base-config',
        help='path to governance.yml at the protected base ref',
    )
    parser.add_argument(
        '--base-budget',
        default=None,
        help='Path to the fail-loud budget at the protected base ref.',
    )
    parser.add_argument(
        '--bootstrap',
        action='store_true',
        help='First-commit override: skip the base-vs-head ratchet.',
    )
    parser.add_argument(
        '--update-budget',
        action='store_true',
        help='Regenerate the fail_loud section of .github/budgets.json',
    )
    args = parser.parse_args()

    if args.update_budget:
        update_budget()
        return 0

    if not BUDGET_PATH.exists():
        raise SystemExit(
            f'fail_loud_gate: {BUDGET_PATH.relative_to(REPO_ROOT)} is missing. '
            f'Run with --update-budget to create it.'
        )
    budget_head = _load_json(BUDGET_PATH, 'head budget')

    # Same base-vs-head semantics as typing_gate: exactly one of
    # --base-budget PATH or --bootstrap must be supplied.
    if args.bootstrap and args.base_budget is not None:
        print(
            'fail_loud_gate: --bootstrap and --base-budget are mutually exclusive',
            file=sys.stderr,
        )
        return 2
    budget_base: dict[str, object] | None
    if args.bootstrap:
        budget_base = None
    elif args.base_budget is not None:
        base_path = Path(args.base_budget)
        if not base_path.is_file():
            print(
                'FAIL-LOUD GATE -- FAIL',
                '',
                '  gate: runtime-args',
                f'    - base-ref budget not found at {base_path}. '
                'Either restore it on the base ref or pass --bootstrap '
                'if this commit introduces the budget.',
                '',
                '1 failure(s). Merge blocked.',
                sep='\n',
            )
            return 1
        budget_base = _load_json(base_path, 'base budget')
    else:
        print(
            'fail_loud_gate: neither --base-budget nor --bootstrap provided. '
            'The workflow must fetch the budget from the protected base ref '
            'and pass --base-budget PATH.',
            file=sys.stderr,
        )
        return 2

    failures = gate(budget_head, budget_base, args.base_config)
    if failures:
        print('FAIL-LOUD GATE -- FAIL')
        print()
        for msg in failures:
            print(f'  - {msg}')
        print()
        print(f'{len(failures)} failure(s). Merge blocked.')
        return 1

    print('FAIL-LOUD GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
