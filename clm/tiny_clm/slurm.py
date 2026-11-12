"""Keep a parent interactive allocation out of independent batch jobs."""
from __future__ import annotations

import argparse
import os
import shlex


CLIENT_SETTINGS = {"SLURM_CONF", "SLURM_CONF_SERVER"}
OPTION_PREFIXES = ("SBATCH_", "SRUN_", "SALLOC_")
DEVICE_MASKS = {"CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}
STEP_RESOURCE_OPTIONS = {
    "SLURM_GPUS", "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK",
    "SLURM_GPUS_PER_SOCKET", "SLURM_GPU_BIND", "SLURM_GPU_FREQ",
    "SLURM_GRES", "SLURM_GRES_FLAGS", "SLURM_STEP_GRES",
    "SLURM_TRES_BIND", "SLURM_TRES_PER_TASK", "SLURM_CPUS_PER_GPU",
    "SLURM_MEM_PER_GPU", "SLURM_NTASKS_PER_GPU",
}


def submission_environment(base=None):
    """Filter only the sbatch child; Slurm will supply the NEW allocation's state.

    All job options come from our command line/TINY_CLM configuration. Preserve
    client connection settings, Python/module paths and data/cache locations.
    Never use this function inside an allocated job: its job ID is required there.
    """
    base = os.environ if base is None else base
    return {k: v for k, v in base.items() if k in CLIENT_SETTINGS or
            (not k.startswith(("SLURM_", "SLURMD_", *OPTION_PREFIXES))
             and k not in DEVICE_MASKS)}


def step_environment(base=None):
    """Keep actual job/array IDs and device visibility, discard option overrides."""
    base = os.environ if base is None else base
    result = {k: v for k, v in base.items()
              if not k.startswith(OPTION_PREFIXES) and k not in STEP_RESOURCE_OPTIONS}
    result["SLURM_EXPORT_ENV"] = "ALL"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    resource = parser.add_mutually_exclusive_group(required=True)
    resource.add_argument("--cpu", action="store_true")
    resource.add_argument("--gpu", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not os.environ.get("SLURM_JOB_ID"):
        parser.error("a command and a current Slurm allocation are required")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if cpus < 1:
        parser.error("SLURM_CPUS_PER_TASK must be positive")
    launch = ["srun", "--export=ALL", "--nodes=1", "--ntasks=1",
              f"--cpus-per-task={cpus}", "--kill-on-bad-exit=1",
              "--gpus=1" if args.gpu else "--gres=none", *command]
    print("Launching allocated step:", shlex.join(launch), flush=True)
    os.execvpe(launch[0], launch, step_environment())


if __name__ == "__main__":
    main()
