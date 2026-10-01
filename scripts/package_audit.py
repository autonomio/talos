#!/usr/bin/env python3
"""Assert the complete contributor sdist and the isolated Talos wheel contract."""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Final

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

BANNER: Final[str] = 'PACKAGE AUDIT'
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DIST: Final[Path] = REPO_ROOT / 'dist'
BOUNDED_RE: Final[re.Pattern[str]] = re.compile(r'>=[^,;]+,\s*<[^,;]+|==[^,;]+')
REQUIRED_SDIST_PATHS: Final[frozenset[str]] = frozenset({
    'README.md', 'LICENSE', 'NOTICE', 'CHANGELOG.md', 'CONTRIBUTING.md',
    'SECURITY.md', 'CITATION.cff', 'CITATION.bib', 'pyproject.toml', 'governance.yml',
    'AGENTS.md', 'CLAUDE.md', 'SETUP.md', 'scripts/package_audit.py',
    'docs-site/package-lock.json', 'requirements/ci/dev-env.txt',
})
FORBIDDEN_PARTS: Final[frozenset[str]] = frozenset({
    '.git', 'node_modules', '__pycache__', '.pytest_cache', '.ruff_cache',
    '.docusaurus', '.generated', 'test-results', '.venv',
})
FORBIDDEN_PREFIXES: Final[tuple[str, ...]] = (
    'dist/', 'dist-first/', 'dist-second/', 'build/', 'verification-output/',
    'docs-site/build/', 'htmlcov/',
)


def version_part(spec: str) -> str:
    """Exclude environment-marker comparisons from dependency bounds."""
    return spec.split(';', 1)[0].strip()


def unbounded(specs: list[str]) -> list[str]:
    """Return declarations without an exact pin or a two-ended range."""
    return [spec for spec in specs if not BOUNDED_RE.search(version_part(spec))]


def unbounded_dependencies() -> list[str]:
    """Inspect every declared runtime, optional and build dependency."""
    data = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    project = data['project']
    declared = list(project.get('dependencies', []))
    for extra in project.get('optional-dependencies', {}).values():
        declared.extend(extra)
    declared.extend(data['build-system']['requires'])
    return unbounded(declared)


def source_files() -> list[str]:
    """Enumerate tracked and newly authored files while honoring Git exclusions."""
    result = subprocess.run(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=REPO_ROOT, capture_output=True, check=True,
    )
    return sorted({name.decode() for name in result.stdout.split(b'\0')
                   if name and (REPO_ROOT / name.decode()).is_file()})


def sdist_members(path: Path) -> set[str]:
    """Return source archive paths below its single distribution root."""
    with tarfile.open(path, 'r:gz') as archive:
        return {item.name.split('/', 1)[1] for item in archive if '/' in item.name}


def wheel_members(path: Path) -> set[str]:
    """Return every installed-wheel path."""
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def audit_sdist(path: Path) -> list[str]:
    """Require every authored contributor file and byte-check its archive copy."""
    failures: list[str] = []
    with tarfile.open(path, 'r:gz') as archive:
        files = {item.name.split('/', 1)[1]: item for item in archive
                 if item.isfile() and '/' in item.name}
        expected = set(source_files()) | REQUIRED_SDIST_PATHS
        for name in sorted(expected):
            item = files.get(name)
            if item is None:
                failures.append(f'sdist is missing {name}')
                continue
            stream = archive.extractfile(item)
            if stream is None:
                failures.append(f'sdist cannot read {name}')
                continue
            with stream:
                shipped = hashlib.sha256(stream.read()).digest()
            source = REPO_ROOT / name
            if source.is_file() and shipped != hashlib.sha256(source.read_bytes()).digest():
                failures.append(f'sdist bytes differ from source: {name}')
        for name in sorted(files):
            if FORBIDDEN_PARTS.intersection(Path(name).parts) or name.startswith(FORBIDDEN_PREFIXES):
                failures.append(f'sdist ships generated content: {name}')
    return failures


def audit_wheel(path: Path) -> list[str]:
    """Keep contributor tooling outside the installed runtime package."""
    failures: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith('/'):
                continue
            first = name.split('/', 1)[0]
            if first != 'talos' and not first.endswith('.dist-info'):
                failures.append(f'wheel ships content outside Talos: {name}')
            if FORBIDDEN_PARTS.intersection(Path(name).parts):
                failures.append(f'wheel ships generated content: {name}')
            if name.startswith('talos/'):
                source = REPO_ROOT / name
                if not source.is_file() or archive.read(name) != source.read_bytes():
                    failures.append(f'wheel bytes differ from source: {name}')
    return failures


def main() -> int:
    """Validate one version's two distributions before release."""
    sdists = sorted(DIST.glob('*.tar.gz'))
    wheels = sorted(DIST.glob('*.whl'))
    if len(sdists) != 1 or len(wheels) != 1:
        print(f'{BANNER} -- FAIL: expected one sdist and one wheel in {DIST}', file=sys.stderr)
        return 2
    failures = audit_sdist(sdists[0]) + audit_wheel(wheels[0])
    failures.extend(f'unbounded dependency: {spec}' for spec in unbounded_dependencies())
    if failures:
        print(f'{BANNER} -- FAIL', file=sys.stderr)
        for failure in failures:
            print(f'  - {failure}', file=sys.stderr)
        return 1
    print(f'{BANNER} -- PASS ({len(sdist_members(sdists[0]))} sdist paths; wheel isolated)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
