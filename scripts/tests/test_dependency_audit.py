"""Advisory lookup aliases preserve every installed hash-locked dependency."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from packaging.requirements import Requirement

SCRIPT = Path(__file__).resolve().parents[1] / 'prepare_dependency_audit.py'
SPEC = importlib.util.spec_from_file_location('prepare_dependency_audit', SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
DIGEST = 'a' * 64
CPU_URL = ('https://download.pytorch.org/whl/cpu/'
           'torch-2.13.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=' + DIGEST)


class AdvisoryIdentityContract(unittest.TestCase):
    """Only official CPU build metadata maps to the public Torch release."""

    def test_full_locked_graph_and_cpu_binding_are_retained(self) -> None:
        source = f'numpy==1.26.4\n    --hash=sha256:{DIGEST}\ntorch @ {CPU_URL}\n    --hash=sha256:{DIGEST}\n'
        installed = {'numpy': '1.26.4', 'torch': '2.13.0+cpu'}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lock, requirements, identities = (root / name for name in ('locked.txt', 'audit.txt', 'identities.json'))
            lock.write_text(source)
            with patch.object(AUDIT.metadata, 'version', side_effect=installed.__getitem__):
                AUDIT.prepare(lock, requirements, identities)
            self.assertEqual(requirements.read_text(), 'numpy==1.26.4\ntorch==2.13.0\n')
            proof = json.loads(identities.read_text())
            self.assertEqual(proof['dependency_count'], 2)
            self.assertEqual(proof['lock_sha256'], hashlib.sha256(source.encode()).hexdigest())
            self.assertEqual({item['name'] for item in proof['dependencies']}, set(installed))
            torch = proof['dependencies'][1]
            self.assertEqual(torch['installed_version'], '2.13.0+cpu')
            self.assertEqual(torch['audit_version'], '2.13.0')
            self.assertEqual(torch['locked_requirement'], f'torch @ {CPU_URL}')
            self.assertEqual(torch['locked_sha256'], [DIGEST])

    def test_untrusted_or_other_wheel_builds_cannot_receive_aliases(self) -> None:
        urls = [CPU_URL.replace('download.pytorch.org', 'download.pytorch.org.example'),
                CPU_URL.replace('https:', 'http:'), CPU_URL.replace('%2Bcpu', '%2Bcu130'),
                CPU_URL.replace('#sha256=' + DIGEST, ''),
                CPU_URL.replace('#sha256=' + DIGEST, '#sha256=' + 'b' * 64)]
        with patch.object(AUDIT.metadata, 'version', return_value='2.13.0+cpu'):
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    AUDIT.advisory_identity(Requirement('torch @ ' + url), {DIGEST})

    def test_installed_wheel_version_mismatch_refuses_alias(self) -> None:
        with patch.object(AUDIT.metadata, 'version', return_value='2.14.1+cpu'), self.assertRaises(ValueError):
            AUDIT.advisory_identity(Requirement('torch @ ' + CPU_URL), {DIGEST})

    def test_unpinned_or_uninstalled_dependencies_fail(self) -> None:
        with patch.object(AUDIT.metadata, 'version', return_value='1.26.4'), self.assertRaises(ValueError):
            AUDIT.advisory_identity(Requirement('numpy>=1.26'), {DIGEST})
        with patch.object(AUDIT.metadata, 'version', side_effect=AUDIT.metadata.PackageNotFoundError), \
                self.assertRaises(AUDIT.metadata.PackageNotFoundError):
            AUDIT.advisory_identity(Requirement('numpy==1.26.4'), {DIGEST})

    def test_hashes_and_resolved_platform_are_required(self) -> None:
        for source in ('numpy==1.26.4', 'numpy==1.26.4\n    --hash=sha256:truncated',
                       f'tomli==2.4.1; python_version < "3.11"\n    --hash=sha256:{DIGEST}', ''):
            with self.subTest(source=source), self.assertRaises(ValueError):
                AUDIT.locked_entries(source)

    def test_duplicate_canonical_packages_fail(self) -> None:
        source = f'numpy==1.26.4\n    --hash=sha256:{DIGEST}\nNumPy==1.26.4\n    --hash=sha256:{DIGEST}\n'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lock = root / 'locked.txt'
            lock.write_text(source)
            with patch.object(AUDIT.metadata, 'version', return_value='1.26.4'), self.assertRaises(ValueError):
                AUDIT.prepare(lock, root / 'audit.txt', root / 'identities.json')
            self.assertFalse((root / 'audit.txt').exists())


if __name__ == '__main__':
    unittest.main()
