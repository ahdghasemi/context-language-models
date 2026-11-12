#!/usr/bin/env python3
"""Plan/submit a declared prompt-format diagnostic. No training or grid."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from submit import ROOT, write, snapshot
from screen import resources, DEFAULTS
from tiny_clm.answer_format import FORMAT_ID, TOOL_FORMAT_ID, protocol_identity, protocol_text
from tiny_clm.environment import verify_environment
from tiny_clm.provenance import verify_snapshot
from tiny_clm.protocol_screen import validate_reference, evidence_files
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
    p.add_argument("--intervention", choices=["answer-format", "tool-call-format"], default="answer-format",
        help="tool-call-format requires a completed protocol-validation source; changes only initial tool-call guidance")
    a = p.parse_args(argv)
    campaign, manifest = a.campaign.resolve(), a.source_manifest.resolve()
    if campaign.exists():
        raise FileExistsError("choose a new campaign; preserve previous experiments")
    identifier = TOOL_FORMAT_ID if a.intervention == "tool-call-format" else FORMAT_ID
    source, _, _, _, cfg = validate_reference(manifest, identifier)
    DEFAULTS["protocol_contextbench"] = (4, "64G", "03:00:00" if identifier == TOOL_FORMAT_ID else "02:00:00")
    commands = []
    for name, gpu, parent, kind in PLAN:
        cmd = ["sbatch", "--parsable", "--job-name=tclm-"+name, *resources(name, gpu),
            "--output="+str(campaign/"logs"/(name+"-%j.out")), "--error="+str(campaign/"logs"/(name+"-%j.err"))]
        dep = [f"--dependency={kind}:<{parent}>"] if parent else []
        print(shlex.join(cmd+dep+["<immutable-snapshot>/compression/slurm/job.sbatch", str(campaign/"manifest.json"), name]))
        commands.append(cmd)
    print(json.dumps({"dry_run": not a.submit, "diagnostic_protocol": protocol_identity(identifier),
        "episodes": len(cfg["screen"]["task_names"])*len(cfg["contextbench"]["modes"]),
        "training": False, "reuse_export": str(Path(source["campaign"])/"export"),
        "task_names": cfg["screen"]["task_names"], "max_tokens": cfg["contextbench"]["max_tokens"],
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
    frozen = evidence_files(manifest, source, identifier)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    work = Path(os.environ["TINY_CLM_WORK"])
    snap = snapshot(commit, os.environ.get("TINY_CLM_SNAPSHOTS", str(work/"snapshots")))
    spec = {"stage": "protocol_screen", "commit": commit, "campaign": str(campaign), "config": cfg,
        "source_manifest": str(manifest), "source_files": frozen, "diagnostic_protocol": protocol_identity(identifier),
        "venv": venv, "environment": environment, "snapshot": str(snap), "snapshot_hash": verify_snapshot(snap),
        "resources": {name: resources(name, gpu) for name, gpu, _, _ in PLAN}}
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign/"logs").mkdir()
    for name, value in (("manifest.json", spec), ("config.json", cfg)):
        write(campaign/name, value)
        (campaign/name).chmod(0o444)
    (campaign/"reminder.txt").write_text(protocol_text(identifier))
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
