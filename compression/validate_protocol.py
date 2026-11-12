#!/usr/bin/env python3
"""Plan/submit eight dense episodes and a two-update resource probe; no grid."""
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
from tiny_clm.protocol_validation import development, expanded_config, evidence_files
from tiny_clm.training_memory import probe_config, estimates
from tiny_clm.provenance import verify_snapshot
from tiny_clm.slurm import submission_environment

PLAN = [("validation_check", False, []),
    ("validation_contextbench", True, [("afterok", "validation_check")]),
    # Serialize GPU stages; still measure memory if the behavior stage fails.
    ("validation_memory", True, [("afterok", "validation_check"), ("afterany", "validation_contextbench")]),
    ("validation_report", False, [("afterany", "validation_contextbench"), ("afterany", "validation_memory")])]
DEFAULTS.update(validation_check=(2, "8G", "00:30:00"),
    validation_contextbench=(4, "64G", "06:00:00"), validation_memory=(4, "128G", "02:00:00"),
    validation_report=(2, "8G", "00:15:00"))


def dependency(deps, ids=None):
    return ["--dependency="+",".join(kind+":"+(ids[parent] if ids else "<"+parent+">")
        for kind, parent in deps)] if deps else []


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", required=True, type=Path, help="passing format-development manifest")
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args(argv)
    campaign, manifest = a.campaign.resolve(), a.source_manifest.resolve()
    if campaign.exists():
        raise FileExistsError("choose a new campaign; preserve historical runs")
    dev, source, lock, _ = development(manifest)
    cfg = expanded_config(dev, lock)
    commands = []
    for name, gpu, deps in PLAN:
        command = ["sbatch", "--parsable", "--job-name=tclm-"+name, *resources(name, gpu),
            "--output="+str(campaign/"logs"/(name+"-%j.out")), "--error="+str(campaign/"logs"/(name+"-%j.err"))]
        print(shlex.join(command+dependency(deps)+["<immutable-snapshot>/compression/slurm/job.sbatch",
            str(campaign/"manifest.json"), name]))
        commands.append(command)
    print(json.dumps({"dry_run": not a.submit, "episodes": 8,
        "diagnostic_protocol": protocol_identity(FORMAT_ID), "training_quality_experiment": False,
        "resource_probe": probe_config(cfg), "analytical_memory": estimates(cfg),
        "reuse_export": str(Path(source["campaign"])/"export"), "temporary_checkpoints_deleted": True}, indent=2))
    if not a.submit:
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit intended changes before submission")
    for key in ("TINY_CLM_PARTITION", "TINY_CLM_CPU_PARTITION", "TINY_CLM_GPU_CONSTRAINT"):
        if not os.environ.get(key):
            raise ValueError("set verified resource setting: "+key)
    venv = os.environ.get("TINY_CLM_VENV", dev["venv"])
    environment = verify_environment(venv, dev["environment"])
    print("Installed environment and passing development protocol verified; source campaigns unchanged", flush=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    work = Path(os.environ["TINY_CLM_WORK"])
    snap = snapshot(commit, os.environ.get("TINY_CLM_SNAPSHOTS", str(work/"snapshots")))
    spec = {"stage": "protocol_validation", "commit": commit, "campaign": str(campaign), "config": cfg,
        "development_manifest": str(manifest), "source_files": evidence_files(manifest, dev),
        "diagnostic_protocol": protocol_identity(FORMAT_ID), "memory_config": probe_config(cfg),
        "venv": venv, "environment": environment, "snapshot": str(snap), "snapshot_hash": verify_snapshot(snap),
        "resources": {name: resources(name, gpu) for name, gpu, _ in PLAN}}
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign/"logs").mkdir()
    for name, value in (("manifest.json", spec), ("config.json", cfg)):
        write(campaign/name, value)
        (campaign/name).chmod(0o444)
    (campaign/"reminder.txt").write_text(FORMAT_TEXT)
    (campaign/"reminder.txt").chmod(0o444)
    env = submission_environment()
    env["TINY_CLM_VENV"] = venv
    ids = {}
    for (name, _, deps), command in zip(PLAN, commands):
        job = subprocess.check_output(command+dependency(deps, ids)+[str(snap/"compression/slurm/job.sbatch"),
            str(campaign/"manifest.json"), name], text=True, env=env).strip().split(";", 1)[0]
        if not job.isdigit():
            raise RuntimeError("unexpected sbatch response: "+job)
        ids[name] = job
        write(campaign/"jobs.json", ids)
        print("submitted "+name+": "+job, flush=True)


if __name__ == "__main__":
    main()
