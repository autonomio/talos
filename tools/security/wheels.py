"""Produce reproducible legacy security wheels from authenticated upstream bytes."""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

from tools.security.patches import apply_patch, digest, inventory_digest

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'legacy-backports.json'


def upstream_wheel(entry: dict[str, object], cache: Path) -> bytes:
    """Download an immutable, hash-pinned upstream wheel or verify its cached copy."""
    source = cache / entry['wheel']
    if source.exists():
        content = source.read_bytes()
    else:
        if not entry['url'].startswith('https://files.pythonhosted.org/'):
            raise ValueError('Legacy wheel source must be the pinned official PyPI artifact')
        with urlopen(entry['url'], timeout=30) as response:
            content = response.read(64 * 1024 * 1024 + 1)
        if len(content) > 64 * 1024 * 1024:
            raise ValueError('Legacy upstream wheel exceeds its download bound')
    if digest(content) != entry['sha256']:
        raise ValueError(f"Upstream wheel SHA-256 mismatch: {entry['wheel']}")
    cache.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    return content


def verified_contents(content: bytes, info: str) -> dict[str, bytes]:
    """Verify every upstream RECORD row and refuse unrecorded or duplicate files."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Upstream wheel has duplicate members')
        files = {name: archive.read(name) for name in names}
    record = info + 'RECORD'
    rows = list(csv.reader(io.StringIO(files[record].decode('utf-8'))))
    recorded: set[str] = set()
    for path, checksum, size in rows:
        if path in recorded:
            raise ValueError('Upstream RECORD has duplicate rows')
        recorded.add(path)
        if path == record and not checksum and not size:
            continue
        algorithm, expected = checksum.split('=', 1)
        actual = base64.urlsafe_b64encode(bytes.fromhex(digest(files[path]))).decode().rstrip('=')
        if algorithm != 'sha256' or actual != expected or len(files[path]) != int(size):
            raise ValueError(f'Upstream RECORD content mismatch: {path}')
    if recorded != set(files):
        raise ValueError('Upstream RECORD does not cover the complete wheel')
    del files[record]
    return files


def security_wheel(entry: dict[str, object], cache: Path, output: Path) -> dict[str, object]:
    """Retain upstream licenses and disclose each backport in installed metadata."""
    name, upstream, version = entry['name'], entry['upstream_version'], entry['patched_version']
    old_info, new_info = f'{name}-{upstream}.dist-info/', f'{name}-{version}.dist-info/'
    files = verified_contents(upstream_wheel(entry, cache), old_info)
    patches = []
    for item in entry['patches']:
        source = files.get(item['path'], b'')
        patch = (ROOT / 'patches' / item['patch']).read_text(encoding='utf-8')
        files[item['path']] = apply_patch(source, patch, item['before_sha256'], item['after_sha256'])
        patches.append({key: value for key, value in item.items() if key != 'patch'})
    files = _versioned_contents(files, name, upstream, version)
    license_record = entry['license']
    license_bytes = (ROOT / license_record['path']).read_bytes()
    if digest(license_bytes) != license_record['sha256']:
        raise ValueError('Owned dependency license differs from its identified upstream source')
    files[new_info + 'LICENSE'] = license_bytes
    proof = {'license': license_record, 'upstream': {'version': upstream, 'wheel': entry['wheel'], 'url': entry['url'],
                          'sha256': entry['sha256']}, 'patches': patches}
    files[new_info + 'AUTONOMIO_PATCHES.json'] = (json.dumps(proof, indent=2) + '\n').encode()
    if inventory_digest(files) != entry['installed_tree_sha256']:
        raise ValueError('Reconstructed security wheel differs from its reviewed complete inventory')
    rows = [[key, 'sha256=' + base64.urlsafe_b64encode(bytes.fromhex(digest(value))).decode().rstrip('='),
             str(len(value))] for key, value in sorted(files.items())]
    rows.append([new_info + 'RECORD', '', ''])
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator='\n').writerows(rows)
    files[new_info + 'RECORD'] = stream.getvalue().encode()
    filename = f'{name}-{version}-py3-none-any.whl'
    target = output / filename
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path, data in sorted(files.items()):
            info = zipfile.ZipInfo(path, date_time=(2024, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    content = buffer.getvalue()
    if digest(content) != entry['output_sha256']:
        raise ValueError('Rebuilt security wheel differs from its reviewed reproducible digest')
    if target.exists() and target.read_bytes() != content:
        raise ValueError(f'Refuse to replace a different security wheel: {filename}')
    output.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return {'name': name, 'upstream_version': upstream, 'patched_version': version,
            'wheel': filename, 'sha256': digest(content), 'patched_files': len(patches),
            'installed_tree_sha256': inventory_digest({path: data for path, data in files.items()
                                                      if path != new_info + 'RECORD'})}


def _versioned_contents(files: dict[str, bytes], name: str, upstream: str, version: str) -> dict[str, bytes]:
    old_info, new_info = f'{name}-{upstream}.dist-info/', f'{name}-{version}.dist-info/'
    files = {new_info + key[len(old_info):] if key.startswith(old_info) else key: value
             for key, value in files.items()}
    metadata = new_info + 'METADATA'
    before, after = f'Version: {upstream}\n'.encode(), f'Version: {version}\n'.encode()
    if files[metadata].count(before) != 1:
        raise ValueError('Upstream metadata version mismatch')
    files[metadata] = files[metadata].replace(before, after)
    if name == 'keras':
        before_python = b'Requires-Python: >=3.9\n'
        if files[metadata].count(before_python) != 1:
            raise ValueError('Upstream Keras Python requirement mismatch')
        files[metadata] = files[metadata].replace(before_python,
            b'Requires-Python: >=3.10.12,!=3.11.0,!=3.11.1,!=3.11.2,!=3.11.3\n')
    init = 'keras/__init__.py' if name == 'keras' else 'google/protobuf/__init__.py'
    for quote in ['"', "'"]:
        files[init] = files[init].replace(f'__version__ = {quote}{upstream}{quote}'.encode(),
                                         f'__version__ = {quote}{version}{quote}'.encode())
    if version.encode() not in files[init]:
        raise ValueError('Patched import version was not updated')
    files[init] = b'# Modified by Autonomio; see AUTONOMIO_PATCHES.json in distribution metadata.\n' + files[init]
    if b'License-File: LICENSE\n' not in files[metadata]:
        files[metadata] = files[metadata].replace(b'\n\n', b'\nLicense-File: LICENSE\n\n', 1)
    return files


def build(cache: Path, output: Path) -> list[dict[str, object]]:
    """Return artifact identities for the complete declared legacy backport set."""
    entries = json.loads(MANIFEST.read_text(encoding='utf-8'))
    receipts = [security_wheel(entry, cache, output) for entry in entries]
    (output / 'receipts.json').write_text(json.dumps(receipts, indent=2) + '\n', encoding='utf-8')
    return receipts


def main() -> int:
    """Expose the reviewed builder without importing any training framework."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.write(json.dumps(build(args.cache, args.output), indent=2) + "\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
