"""Exercise release/configuration refusal paths and distribution content contracts."""
from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]


def load(name: str):
    """Import a standalone contributor script without importing Talos."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load('create_release')
configure = load('configure_repository')
audit = load('package_audit')


class ReleaseContract(unittest.TestCase):
    """A release must agree with source, changelog and selected commit."""

    def test_invalid_tag_is_rejected(self):
        with self.assertRaises(ValueError):
            release.compute_tag('2.0')

    def test_read_version_without_importing_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'pyproject.toml').write_text('[project]\ndynamic=["version"]\n[tool.hatch.version]\npath="version.py"\n')
            (root / 'version.py').write_text('raise RuntimeError("must not import")\n__version__="2.0.1"\n')
            with patch.object(release, 'REPO_ROOT', root):
                self.assertEqual(release.current_version(), '2.0.1')

    def test_changelog_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'CHANGELOG.md').write_text('# v2.0.0\n\n- Add a release.\n')
            with patch.object(release, 'REPO_ROOT', root), self.assertRaises(ValueError):
                release.newest_changelog_section('2.0.1')

    def test_changed_commit_refuses_before_remote_calls(self):
        with patch.object(sys, 'argv', ['create_release', '--tag', 'v2.0.1', '--expected-sha', 'a' * 40]), \
                patch.object(release, 'current_version', return_value='2.0.1'), \
                patch.object(release, 'run', return_value='b' * 40) as commands, \
                self.assertRaises(SystemExit):
            release.main()
        self.assertEqual(commands.call_args.args, ('git', 'rev-parse', 'HEAD'))

    def test_preview_never_tags_or_calls_github(self):
        with patch.object(sys, 'argv', ['create_release', '--tag', 'v2.0.1', '--expected-sha', 'a' * 40]), \
                patch.object(release, 'current_version', return_value='2.0.1'), \
                patch.object(release, 'newest_changelog_section', return_value='- Add a release.'), \
                patch.object(release, 'run', return_value='a' * 40) as commands, \
                redirect_stdout(StringIO()):
            self.assertEqual(release.main(), 0)
        self.assertEqual(commands.call_count, 1)


class ConfigurationContract(unittest.TestCase):
    """Preview and label reconciliation cannot delete unrelated repository state."""

    def test_label_reconciliation_only_upserts(self):
        labels = [{'name': 'slice', 'color': '0052CC', 'description': 'Implementation scope'}]
        with patch.object(configure, 'gh') as api:
            configure.configure_labels('autonomio/talos', labels)
        self.assertEqual(api.call_count, 1)
        self.assertEqual(api.call_args.args[:3], ('label', 'create', 'slice'))
        self.assertNotIn('DELETE', api.call_args.args)

    def test_preview_does_not_call_github(self):
        with patch.object(sys, 'argv', ['configure_repository']), patch.object(configure, 'gh') as api, \
                redirect_stdout(StringIO()):
            self.assertEqual(configure.main(), 0)
        api.assert_not_called()


class DistributionContract(unittest.TestCase):
    """The sdist carries its declared sources and the wheel carries runtime only."""

    def test_wheel_rejects_contributor_and_generated_content(self):
        with tempfile.TemporaryDirectory() as folder:
            wheel = Path(folder) / 'candidate.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('scripts/create_release.py', 'pass')
                archive.writestr('talos/__pycache__/runtime.pyc', 'compiled')
            failures = audit.audit_wheel(wheel)
            self.assertTrue(any('outside Talos' in item for item in failures))
            self.assertTrue(any('generated' in item for item in failures))

    def test_sdist_rejects_missing_and_changed_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'required.txt'
            source.write_text('reviewed')
            candidate = root / 'candidate.tar.gz'
            with tarfile.open(candidate, 'w:gz') as archive:
                archive.add(source, arcname='talos-2.0.1/required.txt')
            source.write_text('changed')
            with patch.object(audit, 'REPO_ROOT', root), \
                    patch.object(audit, 'source_files', return_value=['required.txt', 'missing.txt']), \
                    patch.object(audit, 'REQUIRED_SDIST_PATHS', frozenset()):
                failures = audit.audit_sdist(candidate)
            self.assertIn('sdist is missing missing.txt', failures)
            self.assertIn('sdist bytes differ from source: required.txt', failures)


if __name__ == '__main__':
    unittest.main()
