#!/usr/bin/env python3
"""Bind every installed locked dependency to its upstream advisory identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def locked_entries(text: str) -> list[tuple[Requirement, set[str]]]:
    """Read exact platform-resolved entries, refusing unhashed or unpinned input."""
    blocks: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith('#') or line == '--find-links ./dist':
            continue
        if line.startswith((' ', '\t')):
            if not blocks:
                raise ValueError('lock continuation has no requirement')
            blocks[-1].append(line)
        else:
            blocks.append([line])
    entries = []
    for block in blocks:
        requirement = Requirement(block[0].removesuffix(' \\'))
        hashes = set(re.findall(r'--hash=sha256:([0-9a-f]{64})(?:\s|$)', '\n'.join(block)))
        if not hashes or requirement.marker is not None:
            raise ValueError(f'{requirement.name}: require hashes and a platform-resolved lock')
        entries.append((requirement, hashes))
    if not entries:
        raise ValueError('locked dependency graph is empty')
    return entries


def advisory_identity(requirement: Requirement, hashes: set[str]) -> tuple[str, str]:
    """Map only the official PyTorch CPU build suffix to its public release."""
    installed = metadata.version(requirement.name)
    if requirement.url is None:
        pins = list(requirement.specifier)
        if len(pins) != 1 or pins[0].operator != '==' or '*' in pins[0].version:
            raise ValueError(f'{requirement.name}: require one exact version')
        expected = Version(pins[0].version)
        audited = pins[0].version
    else:
        url = urlsplit(requirement.url)
        filename = unquote(Path(url.path).name)
        name, expected, _, _ = parse_wheel_filename(filename)
        digest = re.fullmatch(r'sha256=([0-9a-f]{64})', url.fragment)
        if (canonicalize_name(requirement.name) != 'torch' or name != 'torch'
                or expected.local != 'cpu' or url.scheme != 'https'
                or url.netloc != 'download.pytorch.org' or not url.path.startswith('/whl/cpu/')
                or url.query or digest is None or digest[1] not in hashes):
            raise ValueError('only a hash-bound official PyTorch CPU wheel has an advisory alias')
        audited = expected.public
    if Version(installed) != expected:
        raise ValueError(f'{requirement.name}: installed {installed} differs from locked {expected}')
    return installed, audited


def prepare(lock: Path, requirements: Path, identities: Path,
            identity: Callable[[Requirement, set[str]], tuple[str, str]] = advisory_identity) -> None:
    """Retain the full installed graph and disclose every lookup transformation."""
    source = lock.read_bytes()
    records = []
    names: set[str] = set()
    for requirement, hashes in locked_entries(source.decode('utf-8')):
        name = canonicalize_name(requirement.name)
        if name in names:
            raise ValueError(f'{name}: duplicate locked dependency')
        names.add(name)
        installed, audited = identity(requirement, hashes)
        records.append({'name': name, 'installed_version': installed, 'audit_version': audited,
                        'locked_requirement': str(requirement), 'locked_sha256': sorted(hashes)})
    requirements.write_text(''.join(f"{item['name']}=={item['audit_version']}\n" for item in records),
                            encoding='utf-8')
    identities.write_text(json.dumps({'lock': str(lock), 'lock_sha256': hashlib.sha256(source).hexdigest(),
                                     'dependency_count': len(records), 'dependencies': records},
                                    indent=2) + '\n', encoding='utf-8')


def main() -> int:
    """Prepare advisory lookups after the original hash-locked installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--requirements', type=Path, required=True)
    parser.add_argument('--identities', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.lock, args.requirements, args.identities)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
