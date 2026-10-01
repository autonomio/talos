"""Scientific software citations stay synchronized without invented identifiers."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.citation import render_bibtex

ROOT = Path(__file__).resolve().parents[2]


def _citation_project(tmp_path):
    (tmp_path / 'talos').mkdir()
    for name in ('CITATION.cff', 'CITATION.bib', 'CHANGELOG.md', 'talos/__init__.py'):
        shutil.copyfile(ROOT / name, tmp_path / name)
    return tmp_path


def test_generated_bibliography_matches_authoritative_version():
    result = render_bibtex(ROOT)
    assert result == (ROOT / 'CITATION.bib').read_text()
    assert result.startswith('@misc{Talos_')
    assert 'note = {Version ' in result
    assert 'doi =' not in result
    assert 'year =' not in result
    assert 'commit =' not in result
    assert 'https://github.com/autonomio/talos' in result
    assert 'Kotila, Mikko' in result


def test_citation_rejects_source_and_changelog_version_drift(tmp_path):
    root = _citation_project(tmp_path)
    citation = root / 'CITATION.cff'
    citation.write_text(re.sub(r'^version:.*$', 'version: "0.0.0"', citation.read_text(), flags=re.MULTILINE))
    with pytest.raises(ValueError, match='match package version'):
        render_bibtex(root)
    shutil.copyfile(ROOT / 'CITATION.cff', citation)
    changelog = root / 'CHANGELOG.md'
    changelog.write_text(re.sub(r'^# v[0-9.]+$', '# v0.0.0', changelog.read_text(), count=1, flags=re.MULTILINE))
    with pytest.raises(ValueError, match='match package version'):
        render_bibtex(root)


def test_citation_cli_check_output_and_version(tmp_path):
    root = _citation_project(tmp_path)
    (root / 'tools').mkdir()
    script = root / 'tools/citation.py'
    shutil.copyfile(ROOT / 'tools/citation.py', script)

    def invoke(*arguments):
        return subprocess.run([sys.executable, str(script), *arguments],
                              cwd=root, capture_output=True, text=True, check=False)

    checked = invoke('--check')
    assert checked.returncode == 0, checked.stderr
    assert 'matches CITATION.cff and source version' in checked.stdout
    displayed = invoke()
    assert displayed.returncode == 0, displayed.stderr
    assert displayed.stdout == (root / 'CITATION.bib').read_text()
    version = invoke('--version')
    assert version.returncode == 0, version.stderr
    assert 'Version ' + version.stdout.strip() in displayed.stdout
    generated = root / 'research.bib'
    written = invoke('--output', str(generated))
    assert written.returncode == 0, written.stderr
    assert generated.read_bytes() == (root / 'CITATION.bib').read_bytes()
    (root / 'CITATION.bib').write_text('stale bibliography')
    rejected = invoke('--check')
    assert rejected.returncode != 0
    assert 'CITATION.bib is stale' in rejected.stderr
