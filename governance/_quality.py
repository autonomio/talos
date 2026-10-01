"""Match inherited quality debt to exact source evidence without admitting new findings."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from _common import REPO_ROOT, comparison_ref, fail_setup

BASELINE = REPO_ROOT / 'governance' / 'quality-baseline.json'


def finding(kind: str, path: Path, message: str, line: int = 0) -> dict[str, str]:
    """Identify one finding by its rule, path and unchanged source evidence."""
    source = path.read_text(encoding='utf-8')
    if path == REPO_ROOT / 'talos' / '__init__.py' and line == 0:
        source = re.sub(r'(?m)^__version__ = .+$', '__version__ = RELEASE_METADATA', source)
    evidence = source if line == 0 else source.splitlines()[line - 1].strip()
    if kind == 'docstrings':
        node = next(node for node in ast.walk(ast.parse(source))
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == line)
        evidence = ast.dump(node, include_attributes=False)
    return {
        'kind': kind,
        'path': path.relative_to(REPO_ROOT).as_posix(),
        'message': message,
        'source_sha256': hashlib.sha256(evidence.encode()).hexdigest(),
    }


def _key(item: dict[str, str]) -> str:
    return json.dumps(item, sort_keys=True)


def load_baseline() -> list[dict[str, str]]:
    """Read auditable debt evidence; absence grants no exemption."""
    if not BASELINE.exists():
        return []
    data = json.loads(BASELINE.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or not isinstance(data.get('findings'), list):
        fail_setup('QUALITY DEBT GATE', 'malformed quality-baseline.json')
    return data['findings']


def new_findings(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """Consume each committed finding at most once; identical new duplicates block."""
    allowed = Counter(_key(item) for item in load_baseline())
    result = []
    for item in items:
        key = _key(item)
        if allowed[key]:
            allowed[key] -= 1
        else:
            result.append(item)
    return result


def baseline_ratchet(base_ref: str) -> list[str]:
    """Forbid new debt evidence; first adoption must prove the inherited source."""
    reachable = subprocess.run(['git', 'rev-parse', '--verify', f'{base_ref}^{{commit}}'],
                               cwd=REPO_ROOT, capture_output=True, check=False)
    if reachable.returncode:
        return [f'base ref is unreachable: {base_ref}']
    command = ['git', 'show', f'{base_ref}:governance/quality-baseline.json']
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    head = load_baseline()
    if result.returncode == 0:
        base = json.loads(result.stdout)
        allowed = Counter(_key(item) for item in base['findings'])
        failures = []
        for item in head:
            key = _key(item)
            if allowed[key]:
                allowed[key] -= 1
            else:
                failures.append(f'baseline increased: {item["path"]}: {item["message"]}')
        return failures
    # Introduction grants evidence only for files already present byte-for-byte
    # at the protected base. A missing/deleted baseline cannot exempt new code.
    failures = []
    anchor = comparison_ref(base_ref)
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', anchor, 'HEAD'], cwd=REPO_ROOT, check=False)
    if ancestor.returncode:
        return ['the documented adoption baseline is not an ancestor of HEAD']
    for path in sorted({item['path'] for item in head}):
        inherited = subprocess.run(
            ['git', 'show', f'{anchor}:{path}'], cwd=REPO_ROOT,
            capture_output=True, check=False,
        )
        current = REPO_ROOT / path
        source = current.read_bytes()
        previous = inherited.stdout
        if path == 'talos/__init__.py':
            source = re.sub(rb'(?m)^__version__ = .+$', b'__version__ = RELEASE_METADATA', source)
            previous = re.sub(rb'(?m)^__version__ = .+$', b'__version__ = RELEASE_METADATA', previous)
        if inherited.returncode or previous != source:
            failures.append(f'initial baseline source is not inherited unchanged: {path}')
    return failures


def strict_policy_ratchet(base_ref: str) -> list[str]:
    """Keep strict selectors, exclusions and complexity bounds from weakening."""
    from _common import loads_toml

    failures = []
    for name in ('governance/ruff.toml', 'pyproject.toml'):
        result = subprocess.run(
            ['git', 'show', f'{base_ref}:{name}'], cwd=REPO_ROOT,
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            if name == 'governance/ruff.toml':
                continue
            return [f'cannot read base configuration: {name}']
        base = loads_toml(result.stdout)
        head = loads_toml((REPO_ROOT / name).read_text())
        if name == 'pyproject.toml':
            base = base.get('tool', {}).get('ruff', {})
            head = head.get('tool', {}).get('ruff', {})
            if not head:
                failures.append(f'{name}: Ruff configuration removed')
        if set(head.get('exclude', [])) - set(base.get('exclude', [])):
            # Initial adoption preserves the previously explicit analysis scan;
            # once governance exists the exact root excludes become ratcheted.
            has_gate = subprocess.run(
                ['git', 'cat-file', '-e', f'{base_ref}:governance/ruff.toml'],
                cwd=REPO_ROOT, check=False, capture_output=True,
            ).returncode == 0
            if has_gate:
                failures.append(f'{name}: scan exclusions increased')
        if name != 'governance/ruff.toml':
            for key in ('extend-ignore', 'extend-per-file-ignores'):
                if head.get('lint', {}).get(key) != base.get('lint', {}).get(key):
                    failures.append(f'{name}: inherited lint escape hatch changed: {key}')
            continue
        base_lint, head_lint = base['lint'], head['lint']
        if set(base_lint['select']) - set(head_lint['select']):
            failures.append(f'{name}: strict selectors narrowed')
        if set(head_lint.get('ignore', [])) - set(base_lint.get('ignore', [])):
            failures.append(f'{name}: ignored rules increased')
        original = base_lint.get('per-file-ignores', {})
        for path, rules in head_lint.get('per-file-ignores', {}).items():
            if set(rules) - set(original.get(path, [])):
                failures.append(f'{name}: ignored rules increased for {path}')
        for table, keys in (('mccabe', ('max-complexity',)),
                            ('pylint', ('max-args', 'max-branches', 'max-statements'))):
            for key in keys:
                if head_lint.get(table, {}).get(key, float('inf')) > base_lint[table][key]:
                    failures.append(f'{name}: {table}.{key} raised')
    return failures
