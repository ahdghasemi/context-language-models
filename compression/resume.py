#!/usr/bin/env python3
"""Resubmit selected array tasks against their original snapshot; dry-run by default."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from submit import resource_args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--indices", required=True, help="comma-separated array indices, e.g. 2,5")
    p.add_argument("--from-stage", choices=["train", "export", "evaluate", "profile", "contextbench"], default="train")
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    manifest = Path(a.manifest).resolve()
    spec = json.loads(manifest.read_text())
    ids = sorted(set(int(s) for s in a.indices.split(",")))
    if not ids or min(ids) < 0 or max(ids) >= len(spec["rows"]):
        raise ValueError("invalid array indices")
    campaign = Path(spec["campaign"])
    for i in ids:
        root = campaign/"runs"/spec["rows"][i]["run_id"]
        status = json.loads((root/"status.json").read_text())
        if a.from_stage == "train":
            if status["status"] == "complete" or not (root/"latest.pt").exists():
                raise ValueError("train resume requires an incomplete checkpoint")
        elif status["status"] != "complete":
            raise ValueError("downstream resume requires completed training")
    stages = (["train", "export", "evaluate", "profile", "contextbench"] if a.from_stage == "train"
              else ["export", "evaluate", "profile", "contextbench"] if a.from_stage == "export"
              else [a.from_stage])
    submitted = {}
    for stage in stages:
        cmd = ["sbatch", "--parsable", "--job-name=tclm-resume-"+stage, *resource_args(True, stage),
               "--array="+",".join(map(str, ids))+"%"+os.environ.get("TINY_CLM_CONCURRENCY", "2"),
               "--output="+str(campaign/"logs"/("resume-"+stage+"-%A_%a.out")),
               "--error="+str(campaign/"logs"/("resume-"+stage+"-%A_%a.err"))]
        dependency = "train" if stage == "export" and "train" in submitted else "export" if stage in {"evaluate", "profile", "contextbench"} and "export" in submitted else None
        if dependency:
            cmd += ["--dependency=aftercorr:"+submitted[dependency]]
        if stage == "train":
            cmd += ["--export=ALL,TINY_CLM_RESUME=1"]
        cmd += [str(Path(spec["snapshot"])/"compression/slurm/job.sbatch"), str(manifest), stage]
        print(shlex.join(cmd))
        if a.submit:
            # Do not race an existing owner; the trainer also takes an exclusive lock.
            old_jobs = json.loads((campaign/"jobs.json").read_text())
            for old in old_jobs.values():
                for i in ids:
                    active = subprocess.run(["squeue", "-h", "-j", f"{old}_{i}", "-o", "%i"], capture_output=True, text=True)
                    if active.returncode == 0 and active.stdout.strip():
                        raise RuntimeError(f"original task {old}_{i} is still active")
            submitted[stage] = subprocess.check_output(cmd, text=True).strip().split(";", 1)[0]
        else:
            submitted[stage] = "<"+stage+">"
    if a.submit:
        import time
        (campaign/("resume-jobs-"+str(int(time.time()))+".json")).write_text(json.dumps(submitted, indent=2))
    print("After these jobs finish, rerun aggregate.py on the same manifest to include every run.")


if __name__ == "__main__":
    main()
