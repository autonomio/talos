#!/usr/bin/env python3
"""Audit the complete compiled runtime graph and enforce active, time-boxed exceptions."""
from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
from functools import partial

from _common import REPO_ROOT, TOMLDecodeError, fail_setup, loads_toml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PYPROJECT = REPO_ROOT / 'pyproject.toml'
EXCEPTIONS = REPO_ROOT / '.github' / 'vuln_exceptions.json'

# Bind this gate's banner to the shared setup-failure reporter.
_fail_setup = partial(fail_setup, 'DEPENDENCY VULNERABILITY GATE')


def _runtime_dependencies() -> list[str]:
    try:
        data = loads_toml(PYPROJECT.read_text(encoding='utf-8'))
    except (OSError, TOMLDecodeError) as exc:
        _fail_setup(f'cannot read pyproject.toml: {exc}')
    project = data.get('project', {})
    deps = project.get('dependencies', []) if isinstance(project, dict) else []
    if not isinstance(deps, list):
        _fail_setup('pyproject [project.dependencies] is not a list')
    return [str(d) for d in deps]


def active_exceptions(raw_text: str, today: datetime.date) -> set[str]:
    """Return the set of vulnerability ids with an unexpired, reasoned
    exception. Malformed exception files fail the gate closed."""
    if not raw_text.strip():
        return set()
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        _fail_setup(f'cannot parse {EXCEPTIONS}: {exc}')
    if not isinstance(raw, list):
        _fail_setup(f'{EXCEPTIONS} must be a JSON list of exceptions')
    active: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or not {'id', 'reason', 'expiry'} <= set(item):
            _fail_setup(f'each exception needs id, reason, expiry: {item!r}')
        try:
            expiry = datetime.date.fromisoformat(str(item['expiry']))
        except ValueError:
            _fail_setup(f'exception expiry must be ISO YYYY-MM-DD: {item!r}')
        if expiry >= today and str(item['reason']).strip():
            active.add(str(item['id']))
    return active


def _audit(deps: list[str]) -> list[dict[str, object]]:
    locked = REPO_ROOT / 'requirements/ci/runtime-env.txt'
    if not locked.is_file():
        _fail_setup('missing runtime-env.txt; compile the declared dependencies first')
    pins = {canonicalize_name(name): version for name, version in re.findall(
        r'^([A-Za-z0-9._-]+)==([^\s;]+)', locked.read_text(), re.MULTILINE,
    )}
    for text in deps:
        requirement = Requirement(text)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        version = pins.get(canonicalize_name(requirement.name))
        if version is None or version not in requirement.specifier:
            _fail_setup(f'compiled runtime lock does not satisfy declaration: {text}')
    result = subprocess.run(
        [sys.executable, '-m', 'pip_audit', '-r', str(locked), '--disable-pip', '--no-deps', '--strict',
         '--format', 'json', '--progress-spinner', 'off'],
        check=False, capture_output=True, text=True,
    )
    if result.returncode not in (0, 1):
        _fail_setup(f'pip-audit could not run: {result.stderr.strip() or result.stdout.strip()}')
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        _fail_setup(f'cannot parse pip-audit JSON: {exc}; tool stderr: {result.stderr.strip()}')
    if not isinstance(payload, dict) or not isinstance(payload.get('dependencies'), list):
        _fail_setup('pip-audit JSON must carry the dependency results')
    return payload['dependencies']


def evaluate(audited: list[dict[str, object]], excepted: set[str]) -> list[str]:
    """Turn pip-audit's per-dependency results into blocking findings,
    dropping any vulnerability whose id has an active exception. Pure
    function so the gate's verdict is deterministically testable."""
    findings: list[str] = []
    for dep in audited:
        name = str(dep.get('name', '?'))
        vulns = dep.get('vulns')
        if not isinstance(vulns, list):
            continue
        for vuln in vulns:
            if not isinstance(vuln, dict):
                continue
            vid = str(vuln.get('id', '?'))
            if vid in excepted:
                continue
            fixes = vuln.get('fix_versions')
            fix = ', '.join(str(v) for v in fixes) if isinstance(fixes, list) and fixes else 'none published'
            findings.append(f'{name}: {vid} (fix: {fix})')
    return findings


def main() -> int:
    deps = _runtime_dependencies()
    if not deps:
        print('DEPENDENCY VULNERABILITY GATE -- PASS (no runtime dependencies declared)')
        return 0
    excepted = active_exceptions(
        EXCEPTIONS.read_text(encoding='utf-8') if EXCEPTIONS.is_file() else '',
        datetime.date.today(),
    )
    findings = evaluate(_audit(deps), excepted)
    if findings:
        print('DEPENDENCY VULNERABILITY GATE -- FAIL', file=sys.stderr)
        print('', file=sys.stderr)
        for finding in findings:
            print(f'  {finding}', file=sys.stderr)
        print('', file=sys.stderr)
        print('  Upgrade the dependency, or add a time-boxed entry to', file=sys.stderr)
        print('  .github/vuln_exceptions.json (id + reason + expiry).', file=sys.stderr)
        print(f'{len(findings)} vulnerability(ies). Merge blocked.', file=sys.stderr)
        return 1
    print('DEPENDENCY VULNERABILITY GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
