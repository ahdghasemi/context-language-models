#!/usr/bin/env python3
"""Stdlib-only Slurm planner. Dry-run is the default; --submit opts into jobs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"clm"))
from tiny_clm.provenance import require_gate, snapshot_files, verify_snapshot  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n")


def grid(config, stage):
    if stage == "baseline":
        cfg = copy.deepcopy(config)
        cfg.update(method="original", seed=0)
        return [{"run_id": "original-keep0.8-seed0", "config": cfg}]
    if stage == "ablations":
        # One seed, matched updates. Constant-mu controls are explicitly ablations.
        rows = []
        for tag, method, kind, keep, fixed in [
            ("original", "original", "mlp", .8, None),
            ("dense", "dense_matched", "mlp", .8, None),
            ("continuation", "constrained", "mlp", .8, None),
            ("constant_weak", "constrained", "mlp", .8, config["penalty"]["initial"]),
            ("constant_strong", "constrained", "mlp", .8, config["penalty"]["minimum"]),
            ("down", "constrained", "down", .4, None),
            ("down_recovery", "svd_recovery", "down", .4, None),
            ("attention", "constrained", "attention", .7, None),
            ("attention_recovery", "svd_recovery", "attention", .7, None)]:
            cfg = copy.deepcopy(config)
            cfg.update(method=method, seed=0)
            cfg["selection"] = {"kind": kind, "keep": keep}
            if fixed is not None:
                cfg["ablation_constant_mu"] = fixed
            rows.append({"run_id": "ablation-"+tag+"-seed0", "config": cfg})
        return rows
    seeds = [0] if stage == "pilot" else [0, 1, 2]
    keeps = [.8, .6] if stage == "pilot" else [.8, .6, .4]
    rows = []
    methods = ["original", "dense_matched", "svd", "svd_recovery", "asvd_uniform", "constrained"]
    if stage == "main":
        methods.append("asvd_adaptive")
    for method in methods:
        for seed in ([0] if method in {"original", "svd", "asvd_uniform", "asvd_adaptive"} else seeds):
            for keep in ([.8] if method in {"original", "dense_matched"} else keeps):
                cfg = copy.deepcopy(config)
                cfg.update(method=method, seed=seed)
                cfg["selection"]["keep"] = keep
                rows.append({"run_id": f"{method}-keep{keep:g}-seed{seed}", "config": cfg})
    return rows


def resource_args(gpu, stage):
    args = ["--nodes=1", "--ntasks=1", "--kill-on-invalid-dep=yes", "--signal=USR1@180"]
    small = stage in {"correctness", "smoke", "aggregate"}
    defaults = {"CPUS": "2" if small else os.environ.get("TINY_CLM_CPUS", "8"),
                "MEM": "8G" if small else os.environ.get("TINY_CLM_MEM", "64G"),
                "TIME": "00:30:00" if small else os.environ.get("TINY_CLM_TIME", "12:00:00")}
    if stage == "environment":
        defaults["TIME"] = "00:15:00"
    if stage == "contextbench":
        defaults["TIME"] = "24:00:00"
    for key, flag in [("CPUS", "cpus-per-task"), ("MEM", "mem"), ("TIME", "time")]:
        args += ["--"+flag+"="+os.environ.get("TINY_CLM_"+stage.upper()+"_"+key, defaults[key])]
    partition = os.environ.get("TINY_CLM_PARTITION" if gpu else "TINY_CLM_CPU_PARTITION", "gpu" if gpu else "cpu")
    args += ["--partition="+partition]
    if gpu:
        if os.environ.get("TINY_CLM_GPUS", "1") != "1":
            raise ValueError("only one-GPU execution is implemented; do not allocate unused GPUs")
        args += ["--gpus=1"]
        constraint = os.environ.get("TINY_CLM_GPU_CONSTRAINT", "")
        if constraint:
            args += ["--constraint="+constraint]
    for env, flag in [("TINY_CLM_ACCOUNT", "account"), ("TINY_CLM_QOS", "qos")]:
        if os.environ.get(env):
            args += ["--"+flag+"="+os.environ[env]]
    return args


def make_plan(stage, n):
    if stage == "environment":
        return [("environment", True, None, [])]
    if stage == "correctness":
        return [("correctness", False, None, []), ("smoke", False, None, [("afterok", "correctness")])]
    if stage == "prepare":
        return [("prepare", False, None, [])]
    return [("environment", True, None, []),
            ("correctness", False, None, [("afterok", "environment")]),
            ("smoke", False, None, [("afterok", "correctness")]),
            ("train", True, n, [("afterok", "smoke")]),
            ("export", True, n, [("aftercorr", "train")]),
            ("evaluate", True, n, [("aftercorr", "export")]),
            ("profile", True, n, [("aftercorr", "export")]),
            ("contextbench", True, n, [("aftercorr", "export")]),
            ("aggregate", False, None, [("afterany", "evaluate"), ("afterany", "profile"), ("afterany", "contextbench")])]


def snapshot(commit, parent):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    target = parent/commit
    if target.exists():
        if not (target/"SNAPSHOT_COMMIT").exists() or (target/"SNAPSHOT_COMMIT").read_text().strip() != commit:
            raise ValueError("incomplete snapshot path already exists")
        verify_snapshot(target)
        return target
    with tempfile.TemporaryDirectory(dir=parent) as tmp:
        archive = Path(tmp)/"source.tar"
        subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), commit], cwd=ROOT, check=True)
        contents = Path(tmp)/"source"
        contents.mkdir()
        with tarfile.open(archive) as t:
            t.extractall(contents, filter="data")
        (contents/"SNAPSHOT_COMMIT").write_text(commit+"\n")
        write(contents/"SNAPSHOT_FILES.json", snapshot_files(contents))
        for file in contents.rglob("*"):
            if file.is_file():
                file.chmod(file.stat().st_mode & ~0o222)
        os.rename(contents, target)
    return target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["environment", "correctness", "prepare", "baseline", "pilot", "main", "ablations"])
    p.add_argument("--config", default="compression/configs/pilot.json")
    p.add_argument("--campaign", help="new absolute campaign directory (auto-generated otherwise)")
    p.add_argument("--submit", action="store_true", help="without this, print only; create no snapshots/jobs")
    p.add_argument("--pilot-gate")
    p.add_argument("--baseline-gate")
    a = p.parse_args(argv)
    cfg = read(a.config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    rows = grid(cfg, a.stage) if a.stage in {"baseline", "pilot", "main", "ablations"} else []
    work = Path(os.environ.get("TINY_CLM_WORK", "/work/pi_aghasemi_umass_edu/tiny_clm_work"))
    campaign = Path(a.campaign or (Path(os.environ.get("TINY_CLM_RUNS", work/"campaigns"))/
                    f"{a.stage}-{time.strftime('%Y%m%dT%H%M%S')}-{commit[:8]}" )).resolve()
    plan = make_plan(a.stage, len(rows))
    spec = {"stage": a.stage, "commit": commit, "config": cfg, "rows": rows,
            "assets": str(Path(os.environ.get("TINY_CLM_ASSETS", work/"assets/qwen3-06-wikitext"))/"assets-lock.json"),
            "contextbench_dir": str(work/"contextbench"/hashlib.sha256(json.dumps(cfg["contextbench"], sort_keys=True).encode()).hexdigest()[:16]),
            "contextbench_source": str(Path(os.environ.get("TINY_CLM_CONTEXTBENCH", work/"external/ContextBench"))/cfg["contextbench"]["revision"]),
            "venv": os.environ.get("TINY_CLM_VENV", str(work/"env")), "campaign": str(campaign)}
    ids, displayed = {}, []
    for name, gpu, array, dependencies in plan:
        command = ["sbatch", "--parsable", "--job-name=tclm-"+name, *resource_args(gpu, name),
                   "--output="+str(campaign/"logs"/(name+"-%A_%a.out")),
                   "--error="+str(campaign/"logs"/(name+"-%A_%a.err"))]
        if array:
            command += [f"--array=0-{array-1}%"+os.environ.get("TINY_CLM_CONCURRENCY", "2")]
        if dependencies:
            command += ["--dependency="+",".join(f"{kind}:<{job}>" for kind, job in dependencies)]
        displayed.append({"name": name, "gpu": gpu, "array": array, "dependencies": dependencies, "command": command})
        print(shlex.join(command+["<immutable-snapshot>/compression/slurm/job.sbatch", str(campaign/"manifest.json"), name]))
    print(json.dumps({"campaign": str(campaign), "commit": commit, "runs": len(rows),
                      "run_ids": [r["run_id"] for r in rows], "dry_run": not a.submit}, indent=2))
    if not a.submit:
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("commit all intended changes before submission; uncommitted code cannot be snapshotted")
    if any(x[1] for x in plan) and not os.environ.get("TINY_CLM_GPU_CONSTRAINT"):
        raise ValueError("set TINY_CLM_GPU_CONSTRAINT from Unity sinfo for BF16-capable >=24 GiB hardware")
    if a.stage == "pilot":
        if not a.baseline_gate:
            raise ValueError("pilot requires --baseline-gate from a completed dense baseline campaign")
        gate = read(a.baseline_gate)
        require_gate(gate, cfg, "baseline")
        spec["baseline_gate"] = gate
    if a.stage in {"main", "ablations"}:
        if not a.pilot_gate:
            raise ValueError("main/ablations require --pilot-gate")
        gate = read(a.pilot_gate)
        require_gate(gate, cfg, "pilot")
        spec["pilot_gate"] = gate
    if a.stage in {"baseline", "pilot", "main", "ablations"}:
        if not Path(spec["assets"]).is_file() or not (Path(spec["contextbench_dir"])/"contextbench-lock.json").is_file():
            raise FileNotFoundError("run prepare for this config and inspect its successful completion first")
    venv = Path(spec["venv"])
    if not (venv/"READY.json").exists():
        raise FileNotFoundError("environment not finalized; run setup_unity.sh")
    snap = snapshot(commit, os.environ.get("TINY_CLM_SNAPSHOTS", str(work/"snapshots")))
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign/"logs").mkdir()
    (campaign/"configs").mkdir()
    spec["snapshot"] = str(snap)
    spec["snapshot_hash"] = verify_snapshot(snap)
    spec["environment"] = read(venv/"READY.json")
    for row in rows:
        target = campaign/"configs"/(row["run_id"]+".json")
        write(target, row["config"])
        target.chmod(0o444)
    write(campaign/"config.json", cfg)
    write(campaign/"manifest.json", spec)
    (campaign/"manifest.json").chmod(0o444)
    for job in displayed:
        command = [x for x in job["command"] if not x.startswith("--dependency=")]
        if job["dependencies"]:
            command += ["--dependency="+",".join(f"{kind}:{ids[name]}" for kind, name in job["dependencies"])]
        command += [str(snap/"compression/slurm/job.sbatch"), str(campaign/"manifest.json"), job["name"]]
        jid = subprocess.check_output(command, text=True).strip().split(";", 1)[0]
        if not jid.isdigit():
            raise RuntimeError(f"unexpected sbatch job ID: {jid}")
        ids[job["name"]] = jid
        write(campaign/"jobs.json", ids)  # preserve partial submission if a later sbatch fails
        print(f"submitted {job['name']}: {jid}")


if __name__ == "__main__":
    main()
