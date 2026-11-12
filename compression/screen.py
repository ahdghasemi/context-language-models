#!/usr/bin/env python3
"""Submit a bounded inference-only backbone screen. Default: print jobs only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from submit import ROOT, read, write, snapshot, resource_args
from tiny_clm.environment import verify_environment
from tiny_clm.provenance import verify_snapshot
from tiny_clm.screen_contract import validate_screen
from tiny_clm.slurm import submission_environment


PLAN = [("screen_check", False, None, "afterok"),
        ("screen_prepare", False, "screen_check", "afterok"),
        ("screen_export", True, "screen_prepare", "afterok"),
        ("screen_contextbench", True, "screen_export", "afterok"),
        ("screen_report", False, "screen_contextbench", "afterany")]
DEFAULTS = {"screen_check": (2, "8G", "00:30:00"),
            "screen_prepare": (4, "32G", "02:00:00"),
            "screen_export": (4, "64G", "01:00:00"),
            "screen_contextbench": (4, "64G", "02:00:00"),
            "screen_report": (2, "8G", "00:15:00")}


def resources(name, gpu):
    # These stages have no optimizer state to checkpoint on an advance signal.
    args = [a for a in resource_args(gpu, name) if not a.startswith("--signal=")]
    for key, flag, default in zip(("CPUS", "MEM", "TIME"),
                                 ("cpus-per-task", "mem", "time"), DEFAULTS[name]):
        args = [a for a in args if not a.startswith("--"+flag+"=")]
        args.append("--"+flag+"="+os.environ.get("TINY_CLM_"+name.upper()+"_"+key, str(default)))
    return args


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT/"compression/configs/screen-qwen3-4b.json"))
    p.add_argument("--reference-manifest", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="checks/downloads only; never inference or training")
    a = p.parse_args(argv)
    campaign = Path(a.campaign).resolve()
    if campaign.exists():
        raise FileExistsError("choose a new campaign; preserve existing logs and outputs")
    cfg, ref = read(a.config), read(a.reference_manifest)
    lock_path = Path(ref["contextbench_dir"])/"contextbench-lock.json"
    validate_screen(cfg, ref, read(lock_path))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    work = Path(os.environ.get("TINY_CLM_WORK", "/work/pi_aghasemi_umass_edu/tiny_clm_work"))
    venv = os.environ.get("TINY_CLM_VENV", ref["venv"])
    plan = PLAN[:2] if a.prepare_only else PLAN
    displayed = []
    for name, gpu, parent, kind in plan:
        command = ["sbatch", "--parsable", "--job-name=tclm-"+name, *resources(name, gpu),
                   "--output="+str(campaign/"logs"/(name+"-%j.out")),
                   "--error="+str(campaign/"logs"/(name+"-%j.err"))]
        dependency = [f"--dependency={kind}:<{parent}>"] if parent else []
        print(shlex.join(command+dependency+["<immutable-snapshot>/compression/slurm/job.sbatch",
                                           str(campaign/"manifest.json"), name]))
        displayed.append((name, command, parent, kind))
    print(json.dumps({"dry_run": not a.submit, "campaign": str(campaign), "commit": commit,
        "model": cfg["model"], "episodes": 0 if a.prepare_only else
            len(cfg["screen"]["task_names"])*len(cfg["contextbench"]["modes"]), "training": False}, indent=2))
    if not a.submit:
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit intended changes before creating immutable snapshots")
    for setting in ("TINY_CLM_PARTITION", "TINY_CLM_CPU_PARTITION", "TINY_CLM_GPU_CONSTRAINT"):
        if not os.environ.get(setting):
            raise ValueError("set verified Unity resource setting: "+setting)
    environment = verify_environment(venv, ref["environment"])
    print("Original installed environment verified; no environment or task lock changed", flush=True)
    snap = snapshot(commit, os.environ.get("TINY_CLM_SNAPSHOTS", str(work/"snapshots")))
    batch_env = submission_environment()
    batch_env["TINY_CLM_VENV"] = str(venv)
    spec = {"stage": "backbone_screen", "commit": commit, "config": cfg,
        "campaign": str(campaign), "venv": str(venv), "environment": environment,
        "snapshot": str(snap), "snapshot_hash": verify_snapshot(snap),
        "reference_manifest": str(Path(a.reference_manifest).resolve()),
        "reference_manifest_sha256": hashlib.sha256(Path(a.reference_manifest).read_bytes()).hexdigest(),
        "contextbench_lock": str(lock_path.resolve()),
        "contextbench_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "screen_assets": str(Path(os.environ.get("TINY_CLM_SCREEN_ASSETS",
                           str(work/"assets/qwen3-4b-instruct2507-screen"))).resolve()),
        "resources": {name: resources(name, gpu) for name, gpu, _, _ in plan},
        "submission_environment_removed": sorted(set(os.environ)-set(batch_env))}
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign/"logs").mkdir()
    for filename, value in [("config.json", cfg), ("manifest.json", spec)]:
        write(campaign/filename, value)
        (campaign/filename).chmod(0o444)
    ids = {}
    for name, command, parent, kind in displayed:
        dependency = [f"--dependency={kind}:{ids[parent]}"] if parent else []
        command += dependency+[str(snap/"compression/slurm/job.sbatch"), str(campaign/"manifest.json"), name]
        jid = subprocess.check_output(command, text=True, env=batch_env).strip().split(";", 1)[0]
        if not jid.isdigit():
            raise RuntimeError("unexpected sbatch result: "+jid)
        ids[name] = jid
        write(campaign/"jobs.json", ids)
        print(f"submitted {name}: {jid}", flush=True)


if __name__ == "__main__":
    main()
