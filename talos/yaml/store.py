"""Persist, resolve and verify content-addressed experiment manifests."""
import hashlib
import json
import re
import warnings
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from typing import TypeGuard

from ruamel.yaml.error import YAMLError

from talos.yaml.config import STORE_RELATIVE
from talos.yaml.config import RoundTripYAML
from talos.yaml.config import find_project_root
from talos.yaml.config import is_list
from talos.yaml.config import is_mapping
from talos.yaml.config import round_trip_yaml


SHA256_PREFIX = 'sha256:'
MANIFEST_URI_SCHEME = 'manifest://'
_SHA256_HEX_LENGTH = 64
_YAML_DUMP_WIDTH = 4096


def canonical_manifest_id(yaml_dict: dict[str, Any]) -> str:

    '''
    Compute the content-addressed manifest ID for a YAML dict.

    Strips any ``lineage`` key before hashing so commit-time and run-time
    identifiers remain identical regardless of whether a lineage block has
    been injected.

    Args:
        yaml_dict (dict[str, Any]): Parsed YAML manifest as a plain dict

    Returns:
        str: ``sha256:<64-hex>`` identifier

    '''

    data_no_lineage = {k: v for k, v in yaml_dict.items() if k != 'lineage'}
    canonical = json.dumps(data_no_lineage, sort_keys=True, default=str)
    return f'{SHA256_PREFIX}{hashlib.sha256(canonical.encode()).hexdigest()}'


def short_id(manifest_id: str) -> str:

    '''
    Extract the leading 8 hex characters of a manifest ID for display.

    Args:
        manifest_id (str): Full sha256:<64-hex> manifest ID

    Returns:
        str: The first 8 hex characters of the hash

    '''

    return manifest_id[len(SHA256_PREFIX):len(SHA256_PREFIX) + 8]


def manifest_name(data: dict[str, Any], fallback: str) -> str:

    '''
    Read metadata.name from a parsed manifest, falling back when absent.

    Args:
        data (dict[str, Any]): Parsed manifest mapping
        fallback (str): Value to return when metadata.name is missing

    Returns:
        str: The metadata.name value, or fallback

    '''

    metadata = data.get('metadata')
    if is_mapping(metadata) and isinstance(metadata.get('name'), str):
        return metadata['name']
    return fallback


def _configured_yaml() -> RoundTripYAML:

    '''Create a round-trip YAML instance with the store's dump settings.'''

    return round_trip_yaml(preserve_quotes=True, width=_YAML_DUMP_WIDTH)


def _index_entry(manifest_id: str,
                 name: str,
                 committed_at: str,
                 parent_id: str | None) -> dict[str, Any]:

    '''Create a manifest store index entry record.'''

    return {
        'id': manifest_id,
        'name': name,
        'committed_at': committed_at,
        'parent_id': parent_id,
        'file': f'{manifest_id[len(SHA256_PREFIX):]}.yaml',
    }


def lineage_block(data: dict[str, Any]) -> dict[str, Any]:

    '''Return the lineage mapping, or an empty dict when it is absent or not a mapping.'''

    lineage = data.get('lineage')
    return lineage if is_mapping(lineage) else {}


def _verify_committed(data: object, name: str, expected_id: str) -> None:
    """Verify the stored lineage label and unchanged scientific manifest content."""
    if not is_mapping(data):
        raise ValueError(f"Invalid manifest format in '{name}'")
    if lineage_block(data).get('id') != expected_id:
        raise ValueError(f"Integrity check failed: '{name}' lineage.id does not match expected '{expected_id}'")
    try:
        content_id = canonical_manifest_id(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Integrity check failed: '{name}' content cannot be hashed: {exc}") from exc
    if content_id != expected_id:
        raise ValueError(f"Integrity check failed: '{name}' content does not match expected '{expected_id}'")


def commit_manifest(yaml_path: Path,
                    project_root: Path,
                    parent_id: str | None = None) -> tuple[str, bool]:

    '''
    Verify committed sources, persist content-addressed drafts and update the index.

    Args:
        yaml_path (Path): Path to the source YAML file
        project_root (Path): Root directory of the talos project
        parent_id (str | None): Parent manifest ID for lineage tracking

    Returns:
        tuple[str, bool]: SHA256 manifest ID and whether it was already in the store

    '''

    content = yaml_path.read_text(encoding='utf-8')

    yaml = _configured_yaml()

    try:
        data = yaml.load(content)
    except YAMLError as exc:
        raise ValueError(f"Cannot parse YAML '{yaml_path.name}': {exc}") from exc
    if not is_mapping(data):
        raise ValueError(f"Invalid YAML format in '{yaml_path.name}': expected a mapping")

    store_path = project_root / STORE_RELATIVE
    source_path = yaml_path.resolve()
    if source_path.parent == store_path.resolve():
        _verify_committed(data, source_path.name, f'{SHA256_PREFIX}{source_path.stem}')

    if parent_id is None:
        source_parent = lineage_block(data).get('parent_id')
        if isinstance(source_parent, str):
            parent_id = source_parent

    manifest_id = canonical_manifest_id(data)

    dest = store_path / f'{manifest_id[len(SHA256_PREFIX):]}.yaml'

    already_existed = dest.exists()

    if not already_existed:
        name = manifest_name(data, fallback=yaml_path.stem)
        committed_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        lineage: dict[str, str] = {'id': manifest_id, 'committed_at': committed_at}
        if parent_id is not None:
            lineage['parent_id'] = parent_id
        data['lineage'] = lineage
        stream = StringIO()
        yaml.dump(data, stream)
        store_path.mkdir(parents=True, exist_ok=True)
        _ = dest.write_text(stream.getvalue(), encoding='utf-8')
        stored_parent_id = parent_id
    else:
        try:
            existing = yaml.load(dest.read_text(encoding='utf-8'))
        except (OSError, YAMLError) as exc:
            raise ValueError(f"Cannot read committed manifest '{dest.name}': {exc}") from exc
        if not is_mapping(existing):
            raise ValueError(f"Invalid committed manifest format in '{dest.name}': expected a mapping")
        _verify_committed(existing, dest.name, manifest_id)
        name = manifest_name(existing, fallback=dest.stem)
        existing_lineage = lineage_block(existing)
        committed_at = existing_lineage.get('committed_at', '')
        stored_parent_id = existing_lineage.get('parent_id')

    _update_index(store_path, _index_entry(manifest_id, name, committed_at, stored_parent_id))

    return manifest_id, already_existed


def resolve_manifest_uri(uri: str, start: Path) -> tuple[Path, Path]:

    '''Resolve a full or unambiguous short manifest URI after verifying stored content.

    Args:
        uri (str): manifest://sha256:<hex> URI
        start (Path): Directory to start searching for the project root

    Returns:
        tuple[Path, Path]: Committed manifest path and project root

    Raises:
        ValueError: If resolution fails or the stored label/content differs from its hash

    '''

    if not uri.startswith(MANIFEST_URI_SCHEME):
        raise ValueError(f"Not a manifest URI: '{uri}'")

    ref = uri[len(MANIFEST_URI_SCHEME):]
    if not ref.startswith(SHA256_PREFIX):
        raise ValueError(f"Manifest URI must use sha256 scheme: '{uri}'")

    hex_hash = ref[len(SHA256_PREFIX):]

    if not re.fullmatch(r'[0-9a-f]{1,64}', hex_hash):
        raise ValueError(f"Malformed hash in manifest URI: '{uri}'")

    project_root = find_project_root(start)
    if project_root is None:
        raise ValueError('No talos project found. Run from inside a project directory.')

    store_path = project_root / STORE_RELATIVE

    if len(hex_hash) == _SHA256_HEX_LENGTH:
        candidate = store_path / f'{hex_hash}.yaml'
        if not candidate.exists():
            raise ValueError(f"Manifest not found in store: '{uri}'")
    else:
        matches = list(store_path.glob(f'{hex_hash}*.yaml'))
        if not matches:
            raise ValueError(f"Manifest not found in store: '{uri}'")
        if len(matches) > 1:
            shorts = ', '.join(p.stem[:12] for p in matches)
            raise ValueError(f"Ambiguous short hash '{hex_hash}' matches multiple manifests: {shorts}")
        candidate = matches[0]

    full_hex = candidate.stem

    yaml_obj = round_trip_yaml()
    try:
        data = yaml_obj.load(candidate.read_text(encoding='utf-8'))
    except (OSError, YAMLError) as exc:
        raise ValueError(f"Cannot read manifest '{candidate.name}': {exc}") from exc
    expected_id = f'{SHA256_PREFIX}{full_hex}'
    _verify_committed(data, candidate.name, expected_id)

    return candidate, project_root


def is_full_manifest_id(value: Any) -> TypeGuard[str]:

    '''
    Check whether a value is a well-formed sha256:<64-hex> manifest ID.

    Args:
        value (Any): Candidate manifest ID

    Returns:
        TypeGuard[str]: True if value is a full sha256 manifest ID

    '''

    if not isinstance(value, str) or not value.startswith(SHA256_PREFIX):
        return False
    hex_part = value[len(SHA256_PREFIX):]
    return len(hex_part) == _SHA256_HEX_LENGTH and all(c in '0123456789abcdef' for c in hex_part)


def normalize_manifest_ref(ref: str) -> str:

    '''
    Normalize a user-supplied manifest reference to a manifest:// URI.

    Accepts a bare hash, a sha256:<hash> reference, or a full
    manifest://sha256:<hash> URI and returns the canonical URI form.

    Args:
        ref (str): User-supplied manifest reference

    Returns:
        str: A manifest://sha256:<hash> URI

    '''

    ref = ref.strip()
    if ref.startswith(MANIFEST_URI_SCHEME):
        return ref
    if ref.startswith(SHA256_PREFIX):
        return f'{MANIFEST_URI_SCHEME}{ref}'
    return f'{MANIFEST_URI_SCHEME}{SHA256_PREFIX}{ref}'


def load_index(project_root: Path) -> dict[str, Any]:

    '''
    Read the manifest store index, returning an empty index when absent.

    Args:
        project_root (Path): Root directory of the talos project

    Returns:
        dict[str, Any]: Parsed index with 'version' and 'manifests' keys

    Raises:
        ValueError: If index.json exists but is corrupted or malformed

    '''

    index_path = project_root / STORE_RELATIVE / 'index.json'
    if not index_path.exists():
        return {'version': 1, 'manifests': []}
    try:
        index = json.loads(index_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f"index.json is corrupted: {index_path}") from exc
    if not is_mapping(index) or not isinstance(index.get('manifests'), list):
        raise ValueError(f"index.json has invalid structure: {index_path}")
    return index


def fork_manifest(committed_path: Path, dest: Path, new_name: str) -> str:

    '''
    Create a development-mode working copy of a committed manifest.

    Copies the committed manifest to ``dest``, sets metadata.name to
    ``new_name``, switches metadata.mode to development, and records the
    source manifest as lineage.parent_id so the lineage is preserved on the
    next commit.

    Args:
        committed_path (Path): Path to the committed manifest file
        dest (Path): Destination path for the working copy
        new_name (str): Value to set for metadata.name

    Returns:
        str: The parent manifest ID recorded in the working copy

    Raises:
        FileExistsError: If dest already exists
        ValueError: If the committed manifest has no metadata mapping

    '''

    if dest.exists():
        raise FileExistsError(str(dest))

    parent_id = f'{SHA256_PREFIX}{committed_path.stem}'

    yaml = _configured_yaml()
    data = yaml.load(committed_path.read_text(encoding='utf-8'))
    if not is_mapping(data):
        raise ValueError(f"Invalid manifest format in '{committed_path.name}'")

    _verify_committed(data, committed_path.name, parent_id)
    metadata = data.get('metadata')
    if not is_mapping(metadata):
        raise ValueError(f"Committed manifest '{committed_path.name}' has no metadata block")

    metadata['name'] = new_name
    metadata['mode'] = 'development'
    data['lineage'] = {'parent_id': parent_id}

    stream = StringIO()
    yaml.dump(data, stream)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _ = dest.write_text(stream.getvalue(), encoding='utf-8')

    return parent_id


def rebuild_index(project_root: Path) -> tuple[int, list[str]]:

    '''
    Rebuild manifests/committed/index.json from the committed manifest files.

    The committed *.yaml files are the source of truth; the index is a derived
    cache. Verify lineage IDs and content hashes against filenames, then index
    valid committed manifests. Report and skip unreadable or malformed entries.

    Args:
        project_root (Path): Root directory of the talos project

    Returns:
        tuple[int, list[str]]: (entries_written, warnings) where warnings
            describe skipped or malformed files

    '''

    store_path = project_root / STORE_RELATIVE
    warnings_out: list[str] = []
    entries: list[dict[str, Any]] = []

    yaml = round_trip_yaml()
    for path in sorted(store_path.glob('*.yaml')):
        try:
            data = yaml.load(path.read_text(encoding='utf-8'))
        except (OSError, YAMLError) as exc:
            warnings_out.append(f"{path.name}: cannot read ({type(exc).__name__}) — skipped")
            continue
        expected_id = f'{SHA256_PREFIX}{path.stem}'
        try:
            _verify_committed(data, path.name, expected_id)
        except ValueError as exc:
            warnings_out.append(f'{exc} — skipped')
            continue

        lineage = lineage_block(data)
        entries.append(_index_entry(
            expected_id,
            manifest_name(data, fallback=path.stem),
            lineage.get('committed_at', ''),
            lineage.get('parent_id'),
        ))

    store_path.mkdir(parents=True, exist_ok=True)
    index = {'version': 1, 'manifests': entries}
    _ = (store_path / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')

    return len(entries), warnings_out


def _update_index(store_path: Path, entry: dict[str, Any]) -> None:

    index_path = store_path / 'index.json'
    index: Any
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            warnings.warn(f"index.json is corrupted — reinitializing: {index_path}", stacklevel=2)
            index = {'version': 1, 'manifests': []}
    else:
        index = {'version': 1, 'manifests': []}

    manifests: list[Any]
    raw_manifests = index.get('manifests') if is_mapping(index) else None
    if is_list(raw_manifests):
        manifests = raw_manifests
    else:
        warnings.warn(f"index.json has invalid structure — reinitializing: {index_path}", stacklevel=2)
        index = {'version': 1, 'manifests': []}
        manifests = []

    kept = [m for m in manifests if not (is_mapping(m) and m.get('id') == entry['id'])]
    kept.append(entry)
    index['manifests'] = kept
    _ = index_path.write_text(json.dumps(index, indent=2), encoding='utf-8')
