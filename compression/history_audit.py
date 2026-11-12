#!/usr/bin/env python3
"""Plan/submit four inference-only completions of one saved first-GET prompt."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

from submit import ROOT, write, snapshot
from screen import resources, DEFAULTS
from tiny_clm.environment import verify_environment
from tiny_clm.history_audit import prepare_source, digest, CONTRACT
from tiny_clm.provenance import verify_snapshot
from tiny_clm.slurm import submission_environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    manifest, campaign = args.source_manifest.resolve(), args.campaign.resolve()
    if campaign.exists():
        raise FileExistsError("choose a new diagnostic campaign; preserve prior results")
    source, bundle = prepare_source(manifest)
    DEFAULTS["history_audit"] = (4, "64G", "00:30:00")
    resource = resources("history_audit", True)
    command = ["sbatch", "--parsable", "--job-name=tclm-history_audit", *resource,
        "--output="+str(campaign/"logs/history_audit-%j.out"),
        "--error="+str(campaign/"logs/history_audit-%j.err")]
    print(shlex.join(command+["<immutable-snapshot>/compression/slurm/job.sbatch", str(campaign/"manifest.json"), "history_audit"]))
    print(f"Four completions; {len(bundle['changes'])} verified historical call envelopes changed; no commands executed")
    if not args.submit:
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit intended changes before submission")
    for key in ("TINY_CLM_PARTITION", "TINY_CLM_GPU_CONSTRAINT"):
        if not os.environ.get(key):
            raise ValueError("set verified resource setting: "+key)
    venv = os.environ.get("TINY_CLM_VENV", source["venv"])
    environment = verify_environment(venv, source["environment"])
    print("Installed environment verified; saved source campaign remains unchanged", flush=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    work = Path(os.environ["TINY_CLM_WORK"])
    snap = snapshot(commit, os.environ.get("TINY_CLM_SNAPSHOTS", str(work/"snapshots")))
    spec = {"stage": "history_audit", "contract": CONTRACT, "commit": commit, "campaign": str(campaign),
        "config": source["config"], "source_manifest": str(manifest), "inputs_sha256": digest(bundle),
        "venv": venv, "environment": environment, "snapshot": str(snap), "snapshot_hash": verify_snapshot(snap),
        "resources": resource}
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign/"logs").mkdir()
    for name, value in (("manifest.json", spec), ("config.json", source["config"]), ("replay-inputs.json", bundle)):
        write(campaign/name, value)
        (campaign/name).chmod(0o444)
    env = submission_environment()
    env["TINY_CLM_VENV"] = venv
    job = subprocess.check_output(command+[str(snap/"compression/slurm/job.sbatch"), str(campaign/"manifest.json"),
        "history_audit"], text=True, env=env).strip().split(";", 1)[0]
    if not job.isdigit():
        raise RuntimeError("unexpected sbatch response: "+job)
    write(campaign/"jobs.json", {"history_audit": job})
    print("submitted history_audit: "+job, flush=True)


if __name__ == "__main__":
    main()
