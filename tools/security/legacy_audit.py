"""Retain upstream findings and fail on any unproved legacy dependency disposition."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from scripts.prepare_dependency_audit import advisory_identity, prepare
from tools.security.installed import verified_backport
from tools.security.wheels import MANIFEST

ROOT = Path(__file__).resolve().parent


def legacy_identity(requirement: Requirement, hashes: set[str]) -> tuple[str, str]:
    """Map only fully verified owned legacy patches to their honest upstream release."""
    name = canonicalize_name(requirement.name)
    names = {entry['name'] for entry in json.loads(MANIFEST.read_text())}
    if name in names:
        upstream = verified_backport(name)
        entry = next(item for item in json.loads(MANIFEST.read_text()) if item['name'] == name)
        if (requirement.url is not None or str(requirement.specifier) != '==' + entry['patched_version']
                or hashes != {entry['output_sha256']}):
            raise ValueError('Owned legacy backport differs from its upstream lock entry')
        return metadata.version(requirement.name), upstream
    return advisory_identity(requirement, hashes)


def dispositions(report: dict, identities: dict) -> list[dict[str, str]]:
    """Keep every advisory visible while requiring an exact reviewed repair or absence proof."""
    mapping = json.loads((ROOT / 'legacy-advisories.json').read_text())
    graph = report['dependencies']
    expected = {item['name']: item for item in identities['dependencies']}
    if len(graph) != len(expected) or {item['name'] for item in graph} != set(expected):
        raise ValueError('Legacy audit did not cover the complete installed dependency graph')
    result = []
    for dependency in graph:
        name = dependency['name']
        if dependency.get('version') != expected[name]['audit_version'] or not isinstance(dependency.get('vulns'), list):
            raise ValueError(f'Legacy audit identity or findings missing: {name}')
        if name in {'keras', 'protobuf'}:
            verified_backport(name)
        for vulnerability in dependency['vulns']:
            result.append(_disposition(name, vulnerability, mapping))
    return result


def _disposition(name: str, vulnerability: dict, mapping: list[dict]) -> dict[str, str]:
    identifiers = {vulnerability['id'], *vulnerability.get('aliases', [])}
    matches = [item for item in mapping if item['id'] in identifiers and item['package'] == name]
    if len(matches) != 1:
        raise ValueError(f"Unrepaired legacy advisory: {name}: {vulnerability['id']}")
    entry = matches[0]
    if entry['kind'] == 'absent':
        for source in entry['paths']:
            if Path(metadata.distribution(name).locate_file(source)).exists():
                raise ValueError(f"Advisory absence proof no longer holds: {entry['id']}: {source}")
    elif entry['kind'] != 'repaired':
        raise ValueError('Unsupported legacy advisory disposition')
    return {'package': name, 'id': entry['id'], 'reported_id': vulnerability['id'],
                   'disposition': entry['kind'], 'evidence': entry['evidence']}


def main() -> int:
    """Prepare full upstream lookups, then validate retained real audit output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--requirements', type=Path)
    parser.add_argument('--identities', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.lock is not None:
        if args.requirements is None or args.audit is not None:
            raise ValueError('Preparing legacy audit requires --requirements and excludes --audit')
        prepare(args.lock, args.requirements, args.identities, legacy_identity)
    else:
        if args.audit is None or args.output is None:
            raise ValueError('Validating legacy audit requires --audit and --output')
        if args.execute:
            if args.requirements is None:
                raise ValueError('Executing legacy audit requires --requirements')
            result = subprocess.run([sys.executable, '-m', 'pip_audit', '--strict', '--disable-pip',
                '--no-deps', '-r', str(args.requirements), '--format', 'json', '--output', str(args.audit)],
                check=False)
            if result.returncode not in (0, 1):
                raise ValueError(f'Legacy dependency auditor failed: exit {result.returncode}')
        rows = dispositions(json.loads(args.audit.read_text()), json.loads(args.identities.read_text()))
        args.output.write_text(json.dumps(rows, indent=2) + '\n')
        sys.stdout.write(f'{len(rows)} retained upstream legacy findings have exact verified dispositions\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
