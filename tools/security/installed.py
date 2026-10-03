"""Bind owned backport installations to their upstream identity and repaired bytes."""
from __future__ import annotations

import base64
import json
from importlib import metadata
from pathlib import Path

from tools.security.patches import digest, inventory_digest
from tools.security.wheels import MANIFEST


def verified_backport(name: str) -> str:
    """Return the original audit version only after validating the complete installed wheel."""
    entry = next(item for item in json.loads(MANIFEST.read_text()) if item['name'] == name)
    distribution = metadata.distribution(name)
    if distribution.version != entry['patched_version']:
        raise ValueError(f'{name}: installed dependency is not the declared owned security backport')
    proof = json.loads(distribution.read_text('AUTONOMIO_PATCHES.json') or '{}')
    expected = {'license': entry['license'], 'upstream': {'version': entry['upstream_version'], 'wheel': entry['wheel'],
                            'url': entry['url'], 'sha256': entry['sha256']},
                'patches': [{key: value for key, value in item.items() if key != 'patch'}
                            for item in entry['patches']]}
    if proof != expected:
        raise ValueError(f'{name}: installed backport provenance differs from reviewed source')
    for patch in entry['patches']:
        source = Path(distribution.locate_file(patch['path']))
        if digest(source.read_bytes()) != patch['after_sha256']:
            raise ValueError(f'{name}: installed security patch source differs: {source}')
    inventory = _installed_inventory(distribution, name, entry['patched_version'])
    if inventory_digest(inventory) != entry['installed_tree_sha256']:
        raise ValueError(f'{name}: installed backport complete source inventory differs')
    _verify_record(distribution, name)
    return entry['upstream_version']


def _installed_inventory(distribution: metadata.Distribution, name: str, version: str) -> dict[str, bytes]:
    package = 'keras' if name == 'keras' else 'google/protobuf'
    info = f"{name}-{version}.dist-info"
    inventory = {}
    for prefix in [package, info]:
        directory = Path(distribution.locate_file(prefix))
        for source in directory.rglob('*'):
            if '__pycache__' in source.parts or source.suffix == '.pyc':
                continue
            if source.is_symlink():
                raise ValueError(f'{name}: installed backport contains a source symlink: {source}')
            if not source.is_file():
                continue
            relative = prefix + '/' + source.relative_to(directory).as_posix()
            if relative in {info + '/' + item for item in ['RECORD', 'INSTALLER', 'REQUESTED', 'direct_url.json', 'uv_cache.json']}:
                continue
            inventory[relative] = source.read_bytes()
    return inventory


def _verify_record(distribution: metadata.Distribution, name: str) -> None:
    # RECORD validates all remaining upstream modules and retained license files.
    files = distribution.files
    if files is None:
        raise ValueError(f'{name}: installed distribution has no RECORD')
    for file in files:
        if file.hash is None:
            if str(file).endswith(('/RECORD', '.pyc', '/INSTALLER', '/REQUESTED', '/direct_url.json', '/uv_cache.json')):
                continue
            raise ValueError(f'{name}: unexplained unhashed installed member: {file}')
        source = Path(distribution.locate_file(file))
        actual = base64.urlsafe_b64encode(bytes.fromhex(digest(source.read_bytes()))).decode().rstrip('=')
        if file.hash.mode != 'sha256' or actual != file.hash.value or source.stat().st_size != file.size:
            raise ValueError(f'{name}: installed RECORD content differs: {file}')
