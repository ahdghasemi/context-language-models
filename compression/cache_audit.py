#!/usr/bin/env python3
"""Submit one read-only GPU cache diagnostic against an existing screen export."""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess

from submit import ROOT, read, write, snapshot
from screen import resources, DEFAULTS
from tiny_clm.environment import verify_environment
from tiny_clm.provenance import verify_snapshot, INFERENCE_RUNTIME
from tiny_clm.slurm import submission_environment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args(argv)
    manifest = Path(a.source_manifest).resolve()
    source = read(manifest)
    campaign = Path(a.campaign).resolve()
    if campaign.exists():
        raise FileExistsError("choose a new diagnostic campaign; preserve previous results")
    audit = read(Path(source["campaign"])/"export-audit.json")
    if (source["stage"] != "backbone_screen" or not audit.get("reload_passed") or
            not audit.get("identical_parameters_and_buffers") or audit.get("inference_runtime") != INFERENCE_RUNTIME):
        raise ValueError("requires an existing v2 export with a passed reload audit")
    DEFAULTS["cache_audit"] = (4, "64G", "01:00:00")
    resource = resources("cache_audit", True)
    command = ["sbatch", "--parsable", "--job-name=tclm-cache_audit", *resource,
        "--output="+str(campaign/"logs/cache_audit-%j.out"),
        "--error="+str(campaign/"logs/cache_audit-%j.err")]
    print(shlex.join(command+["<immutable-snapshot>/compression/slurm/job.sbatch", str(campaign/"manifest.json"), "cache_audit"]))
    if not a.submit:
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit intended changes before submission")
    for key in ("TINY_CLM_PARTITION", "TINY_CLM_GPU_CONSTRAINT"):
        if not os.environ.get(key):
            raise ValueError("set verified resource setting: "+key)
    venv = os.environ.get("TINY_CLM_VENV", source["venv"])
    environment = verify_environment(venv, source["environment"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    work = Path(os.environ["TINY_CLM_WORK"])
    snap = snapshot(commit, os.environ.get("TINY_CLM_SNAPSHOTS", str(work/"snapshots")))
    spec = {"stage": "cache_audit", "commit": commit, "config": source["config"], "campaign": str(campaign),
        "venv": venv, "environment": environment, "snapshot": str(snap), "snapshot_hash": verify_snapshot(snap),
        "source_manifest": str(manifest), "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "resources": resource}
    spec["source_files"] = {str(Path(source["campaign"])/name): hashlib.sha256(
        (Path(source["campaign"])/name).read_bytes()).hexdigest()
        for name in ("export-audit.json", "export/compression.json", "prepared.json", "config.json")}
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign/"logs").mkdir()
    for name, value in (("manifest.json", spec), ("config.json", source["config"])):
        write(campaign/name, value)
        (campaign/name).chmod(0o444)
    env = submission_environment()
    env["TINY_CLM_VENV"] = venv
    job = subprocess.check_output(command+[str(snap/"compression/slurm/job.sbatch"), str(campaign/"manifest.json"),
                                          "cache_audit"], text=True, env=env).strip().split(";", 1)[0]
    if not job.isdigit():
        raise RuntimeError("unexpected sbatch response: "+job)
    write(campaign/"jobs.json", {"cache_audit": job})
    print("submitted cache_audit: "+job, flush=True)


if __name__ == "__main__":
    main()
