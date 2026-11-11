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
    retry_log = None
    if a.submit:
        # Include previous retry job IDs, not only the initial submission.
        known = set()
        for record in [campaign/"jobs.json", *campaign.glob("resume-jobs-*.json")]:
            known.update(json.loads(record.read_text()).values())
        active = set(subprocess.check_output(["squeue", "-h", "-r", "-u", os.environ["USER"],
                                              "-o", "%i"], text=True).split())
        collisions = {f"{j}_{i}" for j in known for i in ids} & active
        if collisions:
            raise RuntimeError("selected tasks still active: "+", ".join(sorted(collisions)))
        import time
        retry_log = campaign/("resume-jobs-"+str(time.time_ns())+".json")
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
            jid = subprocess.check_output(cmd, text=True).strip().split(";", 1)[0]
            if not jid.isdigit():
                raise RuntimeError("invalid Slurm job ID: "+jid)
            submitted[stage] = jid
            retry_log.write_text(json.dumps(submitted, indent=2)+"\n")
        else:
            submitted[stage] = "<"+stage+">"
    print("After these jobs finish, rerun aggregate.py on the same manifest to include every run.")


if __name__ == "__main__":
    main()
