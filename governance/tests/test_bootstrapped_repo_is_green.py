"""Administrative file bootstrap preserves the already specialized Talos repository."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_existing_repository_is_not_renamed_or_rebudgeted(tmp_path: Path) -> None:
    repo = tmp_path / 'repo'
    shutil.copytree(REPO_ROOT, repo, ignore=shutil.ignore_patterns(
        '.git', 'node_modules', 'build', '.generated', '.docusaurus', '__pycache__',
        '.venv*', '.pytest_cache', '.ruff_cache', '.hypothesis', 'test-results',
    ))
    paths = ('talos/__init__.py', 'pyproject.toml', '.github/budgets.json',
             'governance/quality-baseline.json', 'README.md')
    before = {path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in paths}
    result = subprocess.run(
        [sys.executable, 'governance/bootstrap_repository.py', '--files-only',
         '--repo-name', 'talos', '--owner', 'autonomio'], cwd=repo,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'already specialized' in result.stdout
    after = {path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in paths}
    assert before == after
    assert (repo / 'talos').is_dir()
