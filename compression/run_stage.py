#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    manifest, stage = Path(sys.argv[1]), sys.argv[2]
    spec = json.loads(manifest.read_text())
    campaign, snapshot = Path(spec["campaign"]), Path(spec["snapshot"])
    from tiny_clm.provenance import verify_snapshot
    verify_snapshot(snapshot, spec["snapshot_hash"])
    if json.loads((campaign/"config.json").read_text()) != spec["config"]:
        raise ValueError("campaign configuration changed after submission")
    from tiny_clm.environment import verify_environment
    verify_environment(spec["venv"], spec["environment"])
    py = [sys.executable, "-m", "tiny_clm.cli"]
    if stage == "environment":
        command = py+["environment", "--require-cuda", "--min-vram-gib", "24", "--output", str(campaign/"environment.json")]
    elif stage == "correctness":
        command = [sys.executable, "-m", "pytest", "-q", "compression/tests", "clm/tests/test_context_string.py",
                   "--junitxml="+str(campaign/"correctness.xml"), "-o", "cache_dir="+str(campaign/"pytest-cache")]
    elif stage == "smoke":
        command = [sys.executable, "-m", "tiny_clm.smoke", "--output", str(campaign/"smoke")]
    elif stage == "prepare":
        subprocess.run(py+["prepare", "--config", str(campaign/"config.json"), "--destination", str(Path(spec["assets"]).parent)], check=True)
        command = [sys.executable, "-m", "tiny_clm.contextbench", "prepare", "--config", str(campaign/"config.json"),
                   "--source", spec["contextbench_source"], "--destination", spec["contextbench_dir"],
                   "--recipe", str(snapshot/"compression/contextbench.def")]
    elif stage.startswith("screen_"):
        command = [sys.executable, "-m", "tiny_clm.screen", str(manifest), stage]
    elif stage == "aggregate":
        command = [sys.executable, str(snapshot/"compression/aggregate.py"), "--manifest", str(manifest)]
    else:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        row = spec["rows"][index]
        run = campaign/"runs"/row["run_id"]
        config = campaign/"configs"/(row["run_id"]+".json")
        if json.loads(config.read_text()) != row["config"]:
            raise ValueError("run configuration changed after submission")
        if stage == "train":
            command = py+["train", "--config", str(config), "--assets", spec["assets"], "--run", str(run)]
            if os.environ.get("TINY_CLM_RESUME") == "1":
                command += ["--resume"]
        elif stage == "export":
            command = py+["export", "--run", str(run)]
        elif stage == "evaluate":
            command = py+["evaluate", "--artifact", str(run/"export"), "--output", str(run/"evaluation.json")]
        elif stage == "profile":
            command = py+["profile", "--artifact", str(run/"export"), "--output", str(run/"profile.json")]
        elif stage == "contextbench":
            command = [sys.executable, "-m", "tiny_clm.contextbench", "run", "--artifact", str(run/"export"),
                       "--config", str(config), "--lock", str(Path(spec["contextbench_dir"])/"contextbench-lock.json"),
                       "--output", str(run/"contextbench")]
        else:
            raise ValueError("unknown stage")
    print("Executing:", command, flush=True)
    # Replace the process so Slurm's advance signal reaches the trainer directly.
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
