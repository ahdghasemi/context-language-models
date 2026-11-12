#!/usr/bin/env python3
"""Submit held-out evaluation only after validation selection is frozen."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

from submit import read, resource_args, write
from tiny_clm.io import digest
from tiny_clm.slurm import submission_environment

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--selection", required=True)
p.add_argument("--submit", action="store_true")
a = p.parse_args()
manifest, selection = Path(a.manifest).resolve(), Path(a.selection).resolve()
spec, chosen = read(manifest), read(selection)
if chosen["manifest_sha256"] != digest(spec):
    raise ValueError("selection does not belong to campaign")
if not chosen["artifacts"]:
    raise ValueError("empty selection")
if a.submit and not os.environ.get("TINY_CLM_GPU_CONSTRAINT"):
    raise ValueError("configure GPU constraint before submission")
root = Path(spec["campaign"])/("final-test-"+digest(chosen)[:16])
jobs = {}
if a.submit and (root/"jobs.json").exists():
    raise FileExistsError("final test was already submitted; inspect jobs/logs before resubmitting individual failed stages")
for stage, gpu, dependency in [("prepare", False, None), ("evaluate", True, "afterok"), ("aggregate", False, "afterany")]:
    command = ["sbatch", "--parsable", "--job-name=tclm-test-"+stage,
               *resource_args(gpu, "contextbench" if gpu else stage),
               "--output="+str(root/(stage+"-%A_%a.out")), "--error="+str(root/(stage+"-%A_%a.err"))]
    if stage == "evaluate":
        command += [f"--array=0-{len(chosen['artifacts'])-1}%"+os.environ.get("TINY_CLM_CONCURRENCY", "2")]
    if dependency:
        parent = "prepare" if stage == "evaluate" else "evaluate"
        command += ["--dependency="+dependency+":"+jobs.get(parent, "<"+parent+">")]
    command += [str(Path(spec["snapshot"])/"compression/slurm/final_test.sbatch"), str(manifest), str(selection), stage]
    print(shlex.join(command))
    if a.submit:
        if gpu and not os.environ.get("TINY_CLM_GPU_CONSTRAINT"):
            raise ValueError("configure GPU constraint before submission")
        root.mkdir(exist_ok=True)
        jid = subprocess.check_output(command, text=True,
                                      env=submission_environment()).strip().split(";", 1)[0]
        if not jid.isdigit():
            raise RuntimeError("invalid Slurm job ID: "+jid)
        jobs[stage] = jid
        write(root/"jobs.json", jobs)
