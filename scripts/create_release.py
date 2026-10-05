#!/usr/bin/env python3
"""Publish the tested Talos version with immutable source and changelog identity."""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
TAG_RE: Final[re.Pattern[str]] = re.compile(r'^v\d+\.\d+\.\d+$')


def run(*args: str) -> str:
    """Return command output or fail with its original diagnostic."""
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def current_version() -> str:
    """Read Hatch's literal version without importing optional frameworks."""
    project = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    static = project['project'].get('version')
    if isinstance(static, str):
        return static
    source = REPO_ROOT / project['tool']['hatch']['version']['path']
    for node in ast.parse(source.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets
        ) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    raise ValueError('Hatch version source has no literal __version__ assignment')


def compute_tag(version: str) -> str:
    """Enforce the repository's release-tag grammar."""
    tag = f'v{version}'
    if TAG_RE.fullmatch(tag) is None:
        raise ValueError(f'invalid release tag: {tag}')
    return tag


def newest_changelog_section(version: str) -> str:
    """Use the newest changelog body verbatim as release notes."""
    lines = (REPO_ROOT / 'CHANGELOG.md').read_text(encoding='utf-8').splitlines()
    headers = [(index, line) for index, line in enumerate(lines) if re.match(r'^# v\S+', line)]
    if not headers or headers[0][1] != f'# v{version}':
        raise ValueError('newest changelog version disagrees with release version')
    start = headers[0][0] + 1
    end = headers[1][0] if len(headers) > 1 else len(lines)
    body = '\n'.join(lines[start:end]).strip()
    if not body:
        raise ValueError('release changelog body is empty')
    return body


def release_exists(repo: str, tag: str) -> bool:
    """Treat only a confirmed 404 as an absent GitHub release."""
    result = subprocess.run(
        ['gh', 'api', f'repos/{repo}/releases/tags/{tag}'],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        return True
    if '(HTTP 404)' in result.stderr:
        return False
    raise RuntimeError(f'cannot inspect release: {result.stderr.strip()}')


def main() -> int:
    """Require an explicit tag and checked commit before any release mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    tag = compute_tag(current_version())
    if args.tag != tag:
        parser.error(f'requested tag {args.tag!r} disagrees with source {tag!r}')
    sha = run('git', 'rev-parse', 'HEAD')
    if args.expected_sha != sha:
        parser.error('release commit changed; review the selected commit again')
    notes = newest_changelog_section(current_version())
    if not args.publish:
        print(f'RELEASE PREVIEW: {tag} at {sha}\n\n{notes}')
        return 0
    repo = os.environ.get('GITHUB_REPOSITORY')
    if repo != 'autonomio/talos':
        parser.error('GITHUB_REPOSITORY must be autonomio/talos')
    if run('git', 'status', '--porcelain'):
        parser.error('release checkout must be clean')
    tagged = run('git', 'tag', '--list', tag)
    if tagged and run('git', 'rev-parse', f'{tag}^{{commit}}') != sha:
        parser.error('existing release tag points to another commit')
    if release_exists(repo, tag):
        if not tagged:
            parser.error('release exists without its selected tag')
        print(f'RELEASE -- SKIP: {tag} already exists at {sha}')
        return 0
    if not tagged:
        run('git', 'tag', '-a', tag, '-m', tag)
        run('git', 'push', 'origin', tag)
    notes_path = REPO_ROOT / 'release-notes.md'
    notes_path.write_text(notes + f'\n\nSource commit: {sha}\n', encoding='utf-8')
    run('gh', 'release', 'create', tag, '--verify-tag', '--title', tag,
        '--notes-file', str(notes_path))
    print(f'RELEASE -- PASS: {tag} at {sha}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
