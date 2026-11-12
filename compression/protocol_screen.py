#!/usr/bin/env python3
"""Plan or submit checks -> two format-reminder episodes -> report. No training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from submit import ROOT, write, snapshot
from screen import resources, DEFAULTS
from tiny_clm.answer_format import FORMAT_ID, FORMAT_TEXT, protocol_identity
from tiny_clm.environment import verify_environment
from tiny_clm.provenance import verify_snapshot
from tiny_clm.protocol_screen import validate_source, evidence_files
from tiny_clm.slurm import submission_environment

PLAN = [("protocol_check", False, None, "afterok"),
        ("protocol_contextbench", True, "protocol_check", "afterok"),
        ("protocol_report", False, "protocol_contextbench", "afterany")]
DEFAULTS.update(protocol_check=(2, "8G", "00:30:00"),
                protocol_contextbench=(4, "64G", "02:00:00"),
                protocol_report=(2, "8G", "00:15:00"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", required=True, type=Path)
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args(argv)
    campaign, manifest = a.campaign.resolve(), a.source_manifest.resolve()
    if campaign.exists():
        raise FileExistsError("choose a new campaign; preserve previous experiments")
    source, _, _, _ = validate_source(manifest)
    commands = []
    for name, gpu, parent, kind in PLAN:
        cmd = ["sbatch", "--parsable", "--job-name=tclm-"+name, *resources(name, gpu),
            "--output="+str(campaign/"logs"/(name+"-%j.out")), "--error="+str(campaign/"logs"/(name+"-%j.err"))]
        dep = [f"--dependency={kind}:<{parent}>"] if parent else []
        print(shlex.join(cmd+dep+["<immutable-snapshot>/compression/slurm/job.sbatch", str(campaign/"manifest.json"), name]))
        commands.append(cmd)
    print(json.dumps({"dry_run": not a.submit, "diagnostic_protocol": protocol_identity(FORMAT_ID),
        "episodes": 2, "training": False, "reuse_export": str(Path(source["campaign"])/"export"),
        "change": "one instruction suffix in both modes; unchanged retrieval, execution, decoding and grading"}, indent=2))
    if not a.submit:
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit intended changes before submission")
    for key in ("TINY_CLM_PARTITION", "TINY_CLM_CPU_PARTITION", "TINY_CLM_GPU_CONSTRAINT"):
        if not os.environ.get(key):
            raise ValueError("set verified resource setting: "+key)
    venv = os.environ.get("TINY_CLM_VENV", source["venv"])
    environment = verify_environment(venv, source["environment"])
    print("Original installed environment verified; source campaign remains unchanged", flush=True)
    frozen = evidence_files(manifest, source)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    work = Path(os.environ["TINY_CLM_WORK"])
    snap = snapshot(commit, os.environ.get("TINY_CLM_SNAPSHOTS", str(work/"snapshots")))
    spec = {"stage": "protocol_screen", "commit": commit, "campaign": str(campaign), "config": source["config"],
        "source_manifest": str(manifest), "source_files": frozen, "diagnostic_protocol": protocol_identity(FORMAT_ID),
        "venv": venv, "environment": environment, "snapshot": str(snap), "snapshot_hash": verify_snapshot(snap),
        "resources": {name: resources(name, gpu) for name, gpu, _, _ in PLAN}}
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign/"logs").mkdir()
    for name, value in (("manifest.json", spec), ("config.json", source["config"])):
        write(campaign/name, value)
        (campaign/name).chmod(0o444)
    (campaign/"reminder.txt").write_text(FORMAT_TEXT)
    (campaign/"reminder.txt").chmod(0o444)
    env = submission_environment()
    env["TINY_CLM_VENV"] = venv
    ids = {}
    for (name, _, parent, kind), command in zip(PLAN, commands):
        dep = [f"--dependency={kind}:{ids[parent]}"] if parent else []
        job = subprocess.check_output(command+dep+[str(snap/"compression/slurm/job.sbatch"),
            str(campaign/"manifest.json"), name], text=True, env=env).strip().split(";", 1)[0]
        if not job.isdigit():
            raise RuntimeError("unexpected sbatch response: "+job)
        ids[name] = job
        write(campaign/"jobs.json", ids)
        print("submitted "+name+": "+job, flush=True)


if __name__ == "__main__":
    main()
