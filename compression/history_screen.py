#!/usr/bin/env python3
"""Submit four fresh CLM episodes: plain-history control versus native history."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from submit import ROOT, write, snapshot
from screen import resources, DEFAULTS
from tiny_clm.environment import verify_environment
from tiny_clm.history_screen import validate_replay, CONTRACT_SCREEN, ARMS
from tiny_clm.provenance import verify_snapshot
from tiny_clm.slurm import submission_environment

PLAN = [("history_check", False, None, "afterok"),
        ("history_contextbench", True, "history_check", "afterok"),
        ("history_report", False, "history_contextbench", "afterany")]
DEFAULTS.update(history_check=(2, "8G", "00:30:00"),
    history_contextbench=(4, "64G", "03:00:00"), history_report=(2, "8G", "00:15:00"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-manifest", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    manifest, campaign = args.replay_manifest.resolve(), args.campaign.resolve()
    if campaign.exists():
        raise FileExistsError("choose a new campaign; preserve existing results")
    source, bundle, files = validate_replay(manifest)
    commands = []
    for name, gpu, parent, kind in PLAN:
        command = ["sbatch", "--parsable", "--job-name=tclm-"+name, *resources(name, gpu),
            "--output="+str(campaign/"logs"/(name+"-%j.out")), "--error="+str(campaign/"logs"/(name+"-%j.err"))]
        dep = [f"--dependency={kind}:<{parent}>"] if parent else []
        print(shlex.join(command+dep+["<immutable-snapshot>/compression/slurm/job.sbatch", str(campaign/"manifest.json"), name]))
        commands.append(command)
    print(json.dumps({"dry_run": not args.submit, "episodes": 4, "modes": ["clm"], "arms": ARMS,
        "task_names": source["config"]["screen"]["task_names"], "training": False,
        "reuse_export": bundle["artifact"], "change": "verified historical call envelopes in inference messages only"}, indent=2))
    if not args.submit:
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit intended changes before submission")
    for key in ("TINY_CLM_PARTITION", "TINY_CLM_CPU_PARTITION", "TINY_CLM_GPU_CONSTRAINT"):
        if not os.environ.get(key):
            raise ValueError("set verified resource setting: "+key)
    venv = os.environ.get("TINY_CLM_VENV", source["venv"])
    environment = verify_environment(venv, source["environment"])
    print("Installed environment and replay evidence verified; source campaigns unchanged", flush=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    work = Path(os.environ["TINY_CLM_WORK"])
    snap = snapshot(commit, os.environ.get("TINY_CLM_SNAPSHOTS", str(work/"snapshots")))
    spec = {"stage": "history_screen", "contract": CONTRACT_SCREEN, "commit": commit,
        "campaign": str(campaign), "config": source["config"], "replay_manifest": str(manifest),
        "source_files": files, "venv": venv, "environment": environment,
        "snapshot": str(snap), "snapshot_hash": verify_snapshot(snap),
        "resources": {name: resources(name, gpu) for name, gpu, _, _ in PLAN}}
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign/"logs").mkdir()
    for name, value in (("manifest.json", spec), ("config.json", spec["config"])):
        write(campaign/name, value)
        (campaign/name).chmod(0o444)
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
