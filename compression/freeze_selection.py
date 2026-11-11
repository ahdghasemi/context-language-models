#!/usr/bin/env python3
"""Freeze chosen deployment artifacts before any final test evaluation."""
import argparse
import copy
from pathlib import Path
from tiny_clm.io import read_json, write_json, digest

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--run-ids", required=True, help="comma-separated validation-selected IDs; include dense controls")
p.add_argument("--rationale", required=True)
p.add_argument("--output", required=True)
p.add_argument("--test-seeds", default="3101,3102,3103", help="new ContextBench evaluation seeds, fixed before test access")
a = p.parse_args()
out = Path(a.output)
if out.exists():
    raise FileExistsError("selection records are immutable; choose a new filename")
spec = read_json(a.manifest)
expected = {r["run_id"] for r in spec["rows"]}
ids = a.run_ids.split(",")
if len(set(ids)) != len(ids):
    raise ValueError("duplicate selected run IDs")
if not set(ids) <= expected:
    raise ValueError("unknown run IDs")
artifacts = []
for rid in ids:
    root = Path(spec["campaign"])/"runs"/rid
    if read_json(root/"evaluation.json")["split"] != "validation":
        raise ValueError("selection must use validation")
    artifact = read_json(root/"export/compression.json")
    if not read_json(root/"contextbench/summary.json")["all_complete"]:
        raise ValueError("selection requires completed CLM validation")
    artifacts.append({"run_id": rid, "path": str(root/"export"), "weight_sha256": artifact["weight_sha256"]})
test_config = copy.deepcopy(spec["config"])
test_seeds = [int(x) for x in a.test_seeds.split(",")]
if not test_seeds or len(set(test_seeds)) != len(test_seeds) or set(test_seeds) & set(test_config["contextbench"]["seeds"]):
    raise ValueError("final ContextBench seeds must be unique and disjoint from validation")
test_config["contextbench"]["seeds"] = test_seeds
write_json(out, {"artifacts": artifacts, "selected_weight_sha256": [r["weight_sha256"] for r in artifacts],
                "rationale": a.rationale, "manifest_sha256": digest(spec), "commit": spec["commit"],
                "test_config": test_config,
                "note": "frozen before test access; retain all validation failures in the report"})
out.chmod(0o444)
