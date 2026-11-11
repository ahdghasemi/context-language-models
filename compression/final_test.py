#!/usr/bin/env python3
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from tiny_clm.io import digest, read_json, write_json
from tiny_clm.evaluate import evaluate
from tiny_clm.provenance import verify_snapshot

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--selection", required=True)
p.add_argument("--stage", choices=["prepare", "evaluate", "aggregate"], required=True)
a = p.parse_args()
spec, selection = read_json(a.manifest), read_json(a.selection)
verify_snapshot(spec["snapshot"], spec["snapshot_hash"])
if selection["manifest_sha256"] != digest(spec):
    raise ValueError("selection is for a different campaign")
if hashlib.sha256(subprocess.check_output([sys.executable, "-m", "pip", "freeze"])).hexdigest() != spec["environment"]["freeze_sha256"]:
    raise ValueError("environment changed")
root = Path(spec["campaign"])/("final-test-"+digest(selection)[:16])
root.mkdir(exist_ok=True)
if a.stage == "prepare":
    from tiny_clm.contextbench import prepare
    prepare(selection["test_config"], spec["contextbench_source"], root/"tasks",
            Path(spec["snapshot"])/"compression/contextbench.def")
elif a.stage == "evaluate":
    from tiny_clm.contextbench import run
    item = selection["artifacts"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
    if read_json(Path(item["path"])/"compression.json")["weight_sha256"] != item["weight_sha256"]:
        raise ValueError("selected artifact changed after selection freeze")
    out = root/item["run_id"]
    out.mkdir(exist_ok=True)
    general = out/"general.json"
    if general.exists():
        if read_json(general)["artifact_sha256"] != item["weight_sha256"]:
            raise ValueError("existing test belongs to another artifact")
    else:
        evaluate(item["path"], general, split="test", final_selection=a.selection, device="cuda")
    run(item["path"], selection["test_config"], root/"tasks/contextbench-lock.json", out/"contextbench")
else:
    results = []
    for item in selection["artifacts"]:
        out = root/item["run_id"]
        record = {"run_id": item["run_id"], "status": "incomplete"}
        if (out/"general.json").exists():
            record["general"] = read_json(out/"general.json")
            if record["general"]["artifact_sha256"] != item["weight_sha256"]:
                raise ValueError("final general metrics identify a different artifact")
        if (out/"contextbench/summary.json").exists():
            record["contextbench"] = read_json(out/"contextbench/summary.json")
            if any(e["weight_sha256"] != item["weight_sha256"] for e in record["contextbench"]["episodes"]):
                raise ValueError("final CLM metrics identify a different artifact")
            if "general" in record and record["contextbench"]["all_complete"]:
                record["status"] = "complete"
        results.append(record)
    write_json(root/"results.json", {"selection": selection, "runs": results})
    if any(r["status"] != "complete" for r in results):
        raise SystemExit("Final test has incomplete/failed runs; see results.json")
