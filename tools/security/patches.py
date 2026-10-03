"""Apply source-identified unified patches without fuzzy matching."""
from __future__ import annotations

import hashlib
import re


def digest(content: bytes) -> str:
    """Identify source bytes before applying or accepting a security change."""
    return hashlib.sha256(content).hexdigest()


def apply_patch(before: bytes, patch: str, before_sha256: str, after_sha256: str) -> bytes:
    """Reject drift, mismatched context and malformed hunk ranges."""
    if digest(before) != before_sha256:
        raise ValueError('Security patch upstream source hash mismatch')
    source = before.decode('utf-8').splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    if len(lines) < 3 or not lines[0].startswith('--- ') or not lines[1].startswith('+++ '):
        raise ValueError('Security patch must contain one unified file diff')
    result: list[str] = []
    cursor = 0
    index = 2
    while index < len(lines):
        start, old_count, new_start, new_count = _hunk_header(lines[index])
        if not cursor <= start <= len(source):
            raise ValueError('Security patch hunks overlap or exceed source')
        result.extend(source[cursor:start])
        if len(result) != new_start:
            raise ValueError('Security patch new hunk position mismatch')
        emitted, cursor, index = _apply_hunk(source, start, lines, index + 1, old_count, new_count)
        result.extend(emitted)
    result.extend(source[cursor:])
    after = ''.join(result).encode('utf-8')
    if digest(after) != after_sha256:
        raise ValueError('Security patch result hash mismatch')
    return after


def _hunk_header(line: str) -> tuple[int, int, int, int]:
    match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n', line)
    if match is None:
        raise ValueError('Security patch has an invalid hunk header')
    return (max(0, int(match[1]) - 1), int(match[2]) if match[2] is not None else 1,
            max(0, int(match[3]) - 1), int(match[4]) if match[4] is not None else 1)


def _apply_hunk(source: list[str], cursor: int, lines: list[str], index: int,
                old_count: int, new_count: int) -> tuple[list[str], int, int]:
    result: list[str] = []
    consumed = 0
    while index < len(lines) and not lines[index].startswith('@@ '):
        line = lines[index]
        operation, content = line[0], line[1:]
        if operation not in {' ', '+', '-'}:
            raise ValueError('Security patch contains an unsupported operation')
        if operation in {' ', '-'}:
            if cursor >= len(source) or source[cursor] != content:
                raise ValueError('Security patch context mismatch')
            cursor += 1
            consumed += 1
        if operation in {' ', '+'}:
            result.append(content)
        index += 1
    if (consumed, len(result)) != (old_count, new_count):
        raise ValueError('Security patch hunk size mismatch')
    return result, cursor, index


def inventory_digest(files: dict[str, bytes]) -> str:
    """Bind an entire source tree independently of its mutable installed RECORD."""
    rows = [path.encode('utf-8') + b'\0' + digest(content).encode('ascii') + b'\n'
            for path, content in sorted(files.items())]
    return digest(b''.join(rows))
