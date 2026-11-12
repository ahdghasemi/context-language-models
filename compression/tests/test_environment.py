"""Real pip/venv tests: source visibility must not hide dependency changes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
from unittest.mock import patch

import pip

from tiny_clm.environment import frozen_packages, safe_diff, verify_environment


class EnvironmentFingerprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="clm-environment-test-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        cls.env = cls.root/"env"
        # Reuse the installed pip implementation offline. CPU nodes need not have
        # the optional ensurepip bootstrap package that was used at initial setup.
        venv.EnvBuilder(with_pip=False).create(cls.env)
        cls.python = cls.env/"bin/python"
        site = Path(subprocess.check_output([str(cls.python), "-I", "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
        shutil.copytree(Path(pip.__file__).parent, site/"pip",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        installed = site/"fixture_clm-0.1.dist-info"
        installed.mkdir()
        metadata = "Metadata-Version: 2.1\nName: fixture-clm\nVersion: 0.1\n"
        (installed/"METADATA").write_text(metadata)
        (installed/"direct_url.json").write_text(json.dumps(
            {"url": "file:///fixture/original-install", "dir_info": {}}))
        cls.source = cls.root/"source"
        cls.source.mkdir()
        egg = cls.source/"fixture_clm.egg-info"
        egg.mkdir()
        (egg/"PKG-INFO").write_text(metadata)
        dependency = site/"fixture_dependency-1.0.dist-info"
        dependency.mkdir()
        cls.dependency_metadata = dependency/"METADATA"

    def setUp(self):
        upgraded = self.dependency_metadata.parent.with_name("fixture_dependency-2.0.dist-info")
        if upgraded.exists():
            shutil.rmtree(upgraded)
        self.dependency_metadata.parent.mkdir(exist_ok=True)
        self.dependency_metadata.write_text(
            "Metadata-Version: 2.1\nName: fixture-dependency\nVersion: 1.0\n")
        self.freeze = frozen_packages(self.python)
        self.ready = {"python": sys.version,
                      "freeze_sha256": hashlib.sha256(self.freeze).hexdigest()}
        (self.env/"freeze.txt").write_bytes(self.freeze)
        (self.env/"READY.json").write_text(json.dumps(self.ready))

    def test_checkout_egg_info_changes_normal_freeze_but_not_isolated_verification(self):
        before = dict(os.environ)
        with patch.dict(os.environ, PYTHONPATH=str(self.source)):
            polluted = subprocess.check_output([str(self.python), "-m", "pip", "freeze"],
                                              cwd=self.source)
            self.assertIn(b"fixture-clm==0.1", polluted)
            self.assertIn(b"fixture-clm @ file:///fixture/original-install", self.freeze)
            self.assertNotEqual(polluted, self.freeze)
            self.assertEqual(verify_environment(self.env, self.ready), self.ready)
        self.assertEqual(dict(os.environ), before)
        self.assertEqual((self.env/"freeze.txt").read_bytes(), self.freeze)
        self.assertEqual(json.loads((self.env/"READY.json").read_text()), self.ready)

    def test_actual_dependency_version_change_is_rejected(self):
        # A real installer changes the dist-info directory version as well.
        upgraded = self.dependency_metadata.parent.with_name("fixture_dependency-2.0.dist-info")
        self.dependency_metadata.parent.rename(upgraded)
        (upgraded/"METADATA").write_text(
            "Metadata-Version: 2.1\nName: fixture-dependency\nVersion: 2.0\n")
        with self.assertRaisesRegex(RuntimeError, "fixture-dependency==2.0"):
            verify_environment(self.env, self.ready)
        self.assertEqual((self.env/"freeze.txt").read_bytes(), self.freeze)

    def test_modified_freeze_record_is_rejected(self):
        (self.env/"freeze.txt").write_bytes(self.freeze+b"unexpected==1\n")
        with self.assertRaisesRegex(RuntimeError, "freeze.txt differs"):
            verify_environment(self.env)

    def test_replaced_ready_record_cannot_override_campaign(self):
        changed = dict(self.ready, freeze_sha256="0"*64)
        (self.env/"READY.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(RuntimeError, "differs from the campaign"):
            verify_environment(self.env, self.ready)

    def test_remote_urls_are_redacted_from_diagnostics(self):
        result = safe_diff(b"pkg==1\n", b"pkg @ https://user:secret@example.test/pkg?token=private\n")
        self.assertIn("<remote URL redacted>", result)
        self.assertNotIn("secret", result)
        self.assertNotIn("private", result)


if __name__ == "__main__":
    unittest.main()
