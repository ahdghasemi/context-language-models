"""Stdlib-only campaign contracts and source snapshot integrity checks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def protocol_contract(config, *, baseline=False):
    cb = config["contextbench"]
    result = {"model": config["model"], "data": config["data"],
              "export_dtype": config["export_dtype"],
              "contextbench": {k: cb[k] for k in ("revision", "budget", "reserve", "max_tokens",
                  "temperature", "top_p", "enable_thinking", "modes", "budget_tokenizer", "calibrate_budget")},
              "thresholds": config["provisional_thresholds"]}
    if baseline:
        result["contextbench"] = dict(cb)
    if not baseline:
        result.update(selection={k: v for k, v in config["selection"].items() if k != "keep"},
                      penalty=config["penalty"],
                      training={k: v for k, v in config["training"].items() if k not in
                          {"stage_steps", "accumulation", "eval_every", "checkpoint_every", "validation_batches"}})
    return result


def require_gate(gate, config, kind):
    if not gate.get("passed") or gate.get("gate_type") != kind:
        raise ValueError(f"a passing {kind} gate is required; run and aggregate that stage")
    expected = protocol_contract(config, baseline=kind == "baseline")
    if gate.get("protocol_contract") != expected:
        raise ValueError(f"{kind} gate uses different model/data revisions or experimental protocol")


def snapshot_files(root):
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()
            and p.name != "SNAPSHOT_FILES.json" and "__pycache__" not in p.parts}


def verify_snapshot(root, expected_hash=None):
    root = Path(root)
    recorded = json.loads((root/"SNAPSHOT_FILES.json").read_text())
    if expected_hash is not None and canonical_hash(recorded) != expected_hash:
        raise ValueError("snapshot manifest differs from campaign")
    if snapshot_files(root) != recorded:
        raise ValueError("source snapshot changed after creation")
    return canonical_hash(recorded)
