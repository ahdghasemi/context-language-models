"""Exercise the Unity mount fix without a GPU or container installation."""
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from tiny_clm.container import container_environment


class ContainerEnvironmentTests(unittest.TestCase):
    def test_spawned_child_has_no_inherited_mounts_but_keeps_runtime_paths(self):
        # Exercise both prefixes, not only the variable present on one Unity node.
        mounts = {f"{prefix}_{suffix}": "/work,/nas,/scratch"
                  for prefix in ("APPTAINER", "SINGULARITY")
                  for suffix in ("BIND", "BINDPATH", "MOUNT")}
        retained = {"APPTAINER_CACHEDIR": "/work/cache", "APPTAINER_TMPDIR": "/work/tmp",
                    "SINGULARITY_CACHEDIR": "/work/cache", "SINGULARITY_TMPDIR": "/work/tmp",
                    "SLURM_JOB_ID": "fixture", "CUDA_VISIBLE_DEVICES": "0"}
        with patch.dict(os.environ, {**mounts, **retained}):
            before = dict(os.environ)
            result = subprocess.run(
                [sys.executable, "-c", "import json,os; print(json.dumps(dict(os.environ)))"],
                env=container_environment(), text=True, capture_output=True, check=True)
            child = json.loads(result.stdout)
            self.assertTrue(all(key not in child for key in mounts))
            self.assertEqual({key: child[key] for key in retained}, retained)
            self.assertEqual(dict(os.environ), before)

    def test_explicit_environment_is_copied_without_unrequested_defaults(self):
        original = {"PATH": "/usr/bin", "APPTAINER_BINDPATH": "/work", "OTHER": "retained"}
        cleaned = container_environment(original)
        self.assertEqual(cleaned, {"PATH": "/usr/bin", "OTHER": "retained"})
        self.assertEqual(original["APPTAINER_BINDPATH"], "/work")
        self.assertEqual(container_environment({}), {})


if __name__ == "__main__":
    unittest.main()
