"""Exercise real process boundaries without Slurm, a GPU or ML dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tiny_clm.slurm import submission_environment, step_environment


ROOT = Path(__file__).resolve().parents[2]


class SlurmEnvironmentTests(unittest.TestCase):
    def test_submission_child_drops_parent_allocation_and_keeps_project_environment(self):
        inherited = {"SLURM_JOB_ID": "parent-job", "SLURM_GPUS_PER_NODE": "a100:1",
                     "SLURM_TRES_PER_TASK": "cpu=4", "SLURM_STEP_GRES": "gpu:a100:1",
                     "SBATCH_GRES": "gpu:a100:1", "SBATCH_EXPORT": "NONE",
                     "SRUN_EXPORT_ENV": "NONE", "SALLOC_GPUS": "1",
                     "SLURMD_NODENAME": "parent-node", "CUDA_VISIBLE_DEVICES": "3"}
        retained = {"TINY_CLM_VENV": "/work/env", "TMPDIR": "/work/tmp",
                    "APPTAINER_CACHEDIR": "/work/cache", "SLURM_CONF": "/site/slurm.conf",
                    "SLURM_CONF_SERVER": "controller", "LD_LIBRARY_PATH": "/site/lib"}
        with patch.dict(os.environ, {**inherited, **retained}):
            before = dict(os.environ)
            process = subprocess.run([sys.executable, "-c",
                "import json,os; print(json.dumps(dict(os.environ)))"],
                env=submission_environment(), capture_output=True, text=True, check=True)
            child = json.loads(process.stdout)
            self.assertTrue(all(k not in child for k in inherited))
            self.assertEqual({k: child[k] for k in retained}, retained)
            self.assertEqual(dict(os.environ), before)

    def test_explicit_environment_does_not_import_ambient_settings(self):
        original = {"SLURM_GPUS": "1", "PROJECT": "keep"}
        self.assertEqual(submission_environment(original), {"PROJECT": "keep"})
        self.assertEqual(original, {"SLURM_GPUS": "1", "PROJECT": "keep"})
        self.assertEqual(submission_environment({}), {})

    def test_step_preserves_new_allocation_identity_and_device_mask(self):
        base = {"SLURM_JOB_ID": "new-job", "SLURM_ARRAY_TASK_ID": "7",
                "SLURM_JOB_GPUS": "2", "SLURM_CPUS_PER_TASK": "2",
                "CUDA_VISIBLE_DEVICES": "0", "SLURM_GPUS_PER_NODE": "a100:1",
                "SLURM_TRES_PER_TASK": "cpu=4,gres/gpu=1", "SRUN_EXPORT_ENV": "NONE"}
        cleaned = step_environment(base)
        for key in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_GPUS",
                    "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES"):
            self.assertEqual(cleaned[key], base[key])
        self.assertNotIn("SLURM_GPUS_PER_NODE", cleaned)
        self.assertNotIn("SLURM_TRES_PER_TASK", cleaned)
        self.assertNotIn("SRUN_EXPORT_ENV", cleaned)
        self.assertEqual(cleaned["SLURM_EXPORT_ENV"], "ALL")

    def launch_fixture(self, gpu):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)/"srun"
            fake.write_text(f"#!{sys.executable}\n"
                            "import json,os,sys\n"
                            "print(json.dumps({'argv':sys.argv[1:], 'env':dict(os.environ)}))\n")
            fake.chmod(0o755)
            env = dict(os.environ, PATH=tmp+os.pathsep+os.environ.get("PATH", ""),
                PYTHONPATH=str(ROOT/"clm"), SLURM_JOB_ID="new-job", SLURM_ARRAY_TASK_ID="0",
                SLURM_CPUS_PER_TASK="2", SLURM_GPUS_PER_NODE="a100:1", SLURM_GPUS="1",
                SLURM_TRES_PER_TASK="cpu=4,gres/gpu=1", SLURM_STEP_GRES="gpu:a100:1",
                SBATCH_GRES="gpu:a100:1", SRUN_EXPORT_ENV="NONE", CUDA_VISIBLE_DEVICES="0")
            process = subprocess.run([sys.executable, "-m", "tiny_clm.slurm",
                "--gpu" if gpu else "--cpu", "--", "worker", "argument with spaces"],
                env=env, capture_output=True, text=True, check=True)
            record = json.loads(process.stdout.splitlines()[-1])
            argv, child = record["argv"], record["env"]
            self.assertEqual(argv[-2:], ["worker", "argument with spaces"])
            self.assertIn("--cpus-per-task=2", argv)
            self.assertIn("--export=ALL", argv)
            self.assertEqual(child["SLURM_JOB_ID"], "new-job")
            self.assertEqual(child["SLURM_ARRAY_TASK_ID"], "0")
            for key in ("SLURM_GPUS_PER_NODE", "SLURM_GPUS", "SLURM_TRES_PER_TASK",
                        "SLURM_STEP_GRES", "SBATCH_GRES", "SRUN_EXPORT_ENV"):
                self.assertNotIn(key, child)
            return argv, child

    def test_cpu_launcher_executes_srun_with_no_gres(self):
        argv, _ = self.launch_fixture(False)
        self.assertIn("--gres=none", argv)
        self.assertFalse(any(x.startswith("--gpus") for x in argv))

    def test_gpu_launcher_executes_srun_with_one_gpu(self):
        argv, child = self.launch_fixture(True)
        self.assertIn("--gpus=1", argv)
        self.assertNotIn("--gres=none", argv)
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "0")


if __name__ == "__main__":
    unittest.main()
