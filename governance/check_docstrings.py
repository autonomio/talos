#!/usr/bin/env python3
"""Docstring conventions gate: the mechanizable, domain-neutral rules.

Enforces, on every function/method docstring in the package, three
deterministic rules:

  - the title verb is not Calculate/Generate/Make/Build;
  - parameter descriptions do not state a default value;
  - a NOTE marker is written 'NOTE:', never 'Note:'/'note:'.

Conventions that depend on a specific data domain are deliberately out of
scope for an app-neutral template; 'title ends with a period' is already
ruff D415.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Final

from _common import REPO_ROOT, exit_if_disabled, gate_setting, resolve_package_dir
from _quality import finding, new_findings

BANNER: Final[str] = 'DOCSTRING GATE'
DEFAULT_FORBIDDEN_TITLE_VERBS: Final[frozenset[str]] = frozenset({
    'calculate', 'generate', 'make', 'build',
})
DEFAULT_IN_DESC_RE: Final[re.Pattern[str]] = re.compile(r'\(default:', re.IGNORECASE)
NOTE_RE: Final[re.Pattern[str]] = re.compile(r'\bnote:', re.IGNORECASE)


def check_docstring(text: str, forbidden: frozenset[str] | None = None) -> list[str]:
    # Returns a list of convention-violation messages for one docstring body.
    issues: list[str] = []
    title = text.strip().split('\n', 1)[0].strip()
    first_word = title.split(' ', 1)[0].lower() if title else ''
    verbs = DEFAULT_FORBIDDEN_TITLE_VERBS if forbidden is None else forbidden
    if first_word in verbs:
        issues.append(
            f"title verb {first_word!r} is forbidden; open with a precise verb "
            f"(e.g. 'Compute', 'Return', 'Parse', 'Validate') that says what the function does"
        )
    if DEFAULT_IN_DESC_RE.search(text):
        issues.append("description states a default value '(default: ...)'; drop it")
    for match in NOTE_RE.finditer(text):
        if match.group(0) != 'NOTE:':
            issues.append(f"note marker {match.group(0)!r} must be written 'NOTE:'")
            break
    return issues


def find_violations(
    source: str, forbidden: frozenset[str] | None = None
) -> list[tuple[int, str, str]]:
    # Returns (lineno, function name, message) for every docstring violation.
    out: list[tuple[int, str, str]] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        for issue in check_docstring(doc, forbidden):
            out.append((node.lineno, node.name, issue))
    return out


def main() -> int:
    exit_if_disabled('docstrings', BANNER)
    source_dir = resolve_package_dir('DOCSTRING CONVENTIONS GATE')
    violations: list[tuple[Path, int, str, str]] = []
    forbidden = frozenset(
        gate_setting('docstrings', 'forbidden_title_verbs',
                     DEFAULT_FORBIDDEN_TITLE_VERBS, BANNER)
    )
    for path in sorted(source_dir.rglob('*.py')):
        for lineno, name, msg in find_violations(path.read_text(encoding='utf-8'), forbidden):
            violations.append((path.relative_to(REPO_ROOT), lineno, name, msg))
    evidence = {id(item): finding('docstrings', REPO_ROOT / item[0], f'{item[2]}: {item[3]}', item[1]) for item in violations}
    new = new_findings(list(evidence.values()))
    violations = [item for item in violations if evidence[id(item)] in new]
    if violations:
        print('DOCSTRING CONVENTIONS GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for rel, lineno, name, msg in violations:
            print(f'  - {rel}:{lineno} {name}: {msg}', file=sys.stderr)
        print('', file=sys.stderr)
        print(f'{len(violations)} violation(s). Merge blocked.', file=sys.stderr)
        return 1
    print('DOCSTRING CONVENTIONS GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
