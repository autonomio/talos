"""The inherited documentation scaffold remains portable and complete."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SITE = REPO_ROOT / 'docs-site'


def _json(name: str) -> dict[str, object]:
    return json.loads((DOCS_SITE / name).read_text(encoding='utf-8'))


def test_product_profile_has_one_portable_identity_boundary() -> None:
    profile = _json('product-docs.json')

    assert profile['productName'] == 'Talos'
    assert profile['organizationName'] == 'Autonomio'
    assert profile['sourceRepoUrl'] == 'https://github.com/autonomio/talos'
    assert profile['sourceBranch'] == 'master'


def test_route_map_owns_five_sections_and_existing_unique_sources() -> None:
    docs_map = _json('docs-map.json')
    sections = docs_map['sections']
    documents = docs_map['documents']

    assert isinstance(sections, list)
    assert [section['label'] for section in sections] == [
        'Overview',
        'Guides',
        'Reference',
        'Developer',
        'Packages',
    ]
    assert isinstance(documents, list)
    sources = [document['source'] for document in documents]
    destinations = [document['dest'] for document in documents]
    routes = [document['slug'] for document in documents]
    assert len(sources) == len(set(sources))
    assert len(destinations) == len(set(destinations))
    assert len(routes) == len(set(routes))
    resolved_root = REPO_ROOT.resolve()
    resolved_sources = [(REPO_ROOT / source).resolve() for source in sources]
    assert all(path.is_relative_to(resolved_root) for path in resolved_sources)
    assert all(path.is_file() for path in resolved_sources)


def test_docs_check_covers_portable_acceptance_surfaces() -> None:
    package = _json('package.json')
    scripts = package['scripts']
    check = scripts['check']

    assert 'lint-markdown.mjs' in scripts['lint']
    assert 'check-external-links.mjs' in scripts['check:external-links']
    assert 'audit-security.mjs' in scripts['security:audit']
    assert 'docusaurus build --no-minify' in check
    assert 'verify:build' in check
    assert 'test:browser' in check


def test_shared_docs_site_has_no_limen_literals() -> None:
    excluded = {
        'node_modules',
        'build',
        '.generated',
        '.docusaurus',
        'test-results',
    }
    files = [
        path
        for path in DOCS_SITE.rglob('*')
        if path.is_file()
        and path.name != 'package-lock.json'
        and not excluded.intersection(path.relative_to(DOCS_SITE).parts)
    ]

    assert all(
        literal not in path.read_text(encoding='utf-8')
        for path in files
        for literal in ('Limen', '/limen/')
    )


def test_seed_identity_is_confined_to_product_configuration() -> None:
    excluded_files = {
        'docs-map.json',
        'product-docs.json',
        'package-lock.json',
    }
    excluded_dirs = {
        'node_modules',
        'build',
        '.generated',
        '.docusaurus',
        'test-results',
    }
    files = [
        path
        for path in DOCS_SITE.rglob('*')
        if path.is_file()
        and path.name not in excluded_files
        and not excluded_dirs.intersection(path.relative_to(DOCS_SITE).parts)
    ]

    assert all(
        literal not in path.read_text(encoding='utf-8')
        for path in files
        for literal in ('new_repository_template', 'new-repository-template')
    )


def test_visual_contract_self_hosts_autonomio_fonts() -> None:
    css = (DOCS_SITE / 'src' / 'css' / 'custom.css').read_text(encoding='utf-8')

    assert "@import '@fontsource/finlandica/latin-400.css';" in css
    assert '#f7f7f2' in css.lower()
    assert '#2a4a70' in css.lower()
    assert 'fonts.googleapis.com' not in css
