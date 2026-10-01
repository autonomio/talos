"""Synchronize the software bibliography with authoritative CFF metadata."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

from ruamel.yaml import YAML

__all__ = ['render_bibtex', 'main']


def _text(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Citation {key} must be a nonempty string')
    return value


def _escape(value: str) -> str:
    replacements = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}',
                    '&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#',
                    '$': r'\$', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(character, character) for character in value)


def _source_version(root: Path) -> str:
    tree = ast.parse((root / 'talos/__init__.py').read_text(encoding='utf-8'))
    versions = [node.value.value for node in tree.body
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and any(isinstance(target, ast.Name) and target.id == '__version__'
                        for target in node.targets)]
    if len(versions) != 1:
        raise ValueError('Package must declare exactly one literal __version__')
    return versions[0]


def render_bibtex(root: Path) -> str:
    """Render source-backed software fields without inferring publication details."""
    with (root / 'CITATION.cff').open(encoding='utf-8') as stream:
        metadata = YAML(typ='safe').load(stream)
    if not isinstance(metadata, dict):
        raise ValueError('CITATION.cff must contain a mapping')
    version = _text(metadata, 'version')
    heading = re.search(r'^# v(\d+\.\d+\.\d+)\s*$',
                        (root / 'CHANGELOG.md').read_text(encoding='utf-8'), re.MULTILINE)
    if version != _source_version(root) or heading is None or version != heading.group(1):
        raise ValueError('CFF version must match package version and newest changelog')
    if metadata.get('cff-version') != '1.2.0' or metadata.get('type') != 'software':
        raise ValueError('Citation must use CFF 1.2.0 and software type')
    authors = metadata.get('authors')
    if not isinstance(authors, list) or not authors:
        raise ValueError('Citation must contain software authors')
    names = []
    for author in authors:
        if not isinstance(author, dict):
            raise ValueError('Citation authors must be mappings')
        names.append(_escape(_text(author, 'family-names')) + ', ' +
                     _escape(_text(author, 'given-names')))
    key = 'Talos_' + version.replace('.', '_')
    fields = [('author', ' and '.join(names)),
              ('title', '{' + _escape(_text(metadata, 'title')) + '}'),
              ('version', _escape(version)),
              ('url', _escape(_text(metadata, 'repository-code'))),
              ('license', _escape(_text(metadata, 'license'))),
              ('note', 'Version ' + _escape(version))]
    lines = ['@misc{' + key + ',']
    lines.extend('  ' + name + ' = {' + value + '},' for name, value in fields)
    lines[-1] = lines[-1].removesuffix(',')
    return '\n'.join([*lines, '}', ''])


def main() -> None:
    """Check synchronization, write an explicit output, or display the version."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check', action='store_true', help='Reject stale CITATION.bib')
    actions.add_argument('--output', type=Path, help='Write generated BibTeX to this path')
    actions.add_argument('--version', action='store_true', help='Show the checked source version')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bibliography = render_bibtex(root)
    if args.version:
        print(_source_version(root))
    elif args.check:
        if (root / 'CITATION.bib').read_bytes() != bibliography.encode('utf-8'):
            raise ValueError('CITATION.bib is stale; regenerate it from CITATION.cff')
        print('CITATION.bib matches CITATION.cff and source version')
    elif args.output:
        args.output.write_text(bibliography, encoding='utf-8', newline='\n')
    else:
        print(bibliography, end='')


if __name__ == '__main__':
    main()
