#!/usr/bin/env python3
"""Audit all expected runs, including failures. Never infer absent measurements."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
from pathlib import Path
import re
import statistics

import numpy as np

from tiny_clm.io import digest, read_json, write_json
from tiny_clm.provenance import protocol_contract, INFERENCE_RUNTIME, LEGACY_INFERENCE_RUNTIME


def validate_episodes(config, summary):
    cb = config["contextbench"]
    if summary["contextbench"] != cb or summary["source_revision"] != cb["revision"]:
        raise ValueError("ContextBench results use different tasks or decoding settings")
    episodes = summary["episodes"]
    runtime = summary.get("inference_runtime", LEGACY_INFERENCE_RUNTIME)
    if any(e.get("inference_runtime", LEGACY_INFERENCE_RUNTIME) != runtime for e in episodes):
        raise ValueError("ContextBench episodes use mixed inference precision policies")
    keys = [(e["episode"], e["mode"]) for e in episodes]
    expected = len(cb["tasks"])*len(cb["seeds"])
    if len(keys) != len(set(keys)) or Counter(e["mode"] for e in episodes) != {m: expected for m in cb["modes"]}:
        raise ValueError("ContextBench episode coverage is incomplete or duplicated")
    if summary["all_complete"] != all(e["status"] == "complete" for e in episodes):
        raise ValueError("ContextBench completion flag contradicts its episodes")


def collect(spec):
    rows, episodes = [], {}
    for item in spec["rows"]:
        cfg, rid = item["config"], item["run_id"]
        root = Path(spec["campaign"])/"runs"/rid
        row = {"run_id": rid, "method": cfg["method"], "seed": cfg["seed"],
               "selected_keep_requested": cfg["selection"]["keep"], "selection_kind": cfg["selection"]["kind"],
               "penalty_schedule": str(cfg["penalty"]), "status": "incomplete"}
        row["ablation_constant_mu"] = cfg.get("ablation_constant_mu")
        missing, identity, runtimes = [], {}, set()
        for filename, key in [("status.json", "training"), ("evaluation.json", "evaluation"),
                              ("deployment_gap.json", "gap"), ("profile.json", "profile"),
                              ("export/compression.json", "artifact"), ("contextbench/summary.json", "clm")]:
            if not (root/filename).exists():
                missing.append(filename)
                continue
            value = read_json(root/filename)
            if key in {"artifact", "evaluation", "profile", "clm", "gap"}:
                runtimes.add(value.get("inference_runtime", LEGACY_INFERENCE_RUNTIME))
            if key in {"evaluation", "profile"}:
                identity[key] = value["artifact_sha256"]
            elif key == "artifact":
                identity[key] = value["weight_sha256"]
            elif key == "clm":
                validate_episodes(cfg, value)
                for i, episode in enumerate(value["episodes"]):
                    identity["episode"+str(i)] = episode["weight_sha256"]
            if key == "training":
                row.update(training_status=value["status"], training_tokens=value["training_tokens"],
                    training_seconds=value["elapsed_seconds"], projection_seconds=value["projection_seconds"])
            elif key == "evaluation":
                if value["split"] != "validation":
                    raise ValueError("main selection aggregation accepts validation metrics only")
                row.update(nll=value["nll"], perplexity=value["perplexity"])
            elif key == "gap":
                row.update(export_nll_delta=value["nll_delta"], reload_error=value["save_reload_max_logit_error"])
            elif key == "artifact":
                row.update(parameters=value["deployment_parameters"], original_parameters=value["original_parameters"],
                           whole_keep=value["whole_keep"], weight_bytes=value["weight_bytes"], weight_sha256=value["weight_sha256"])
            elif key == "profile":
                row["profile_hardware"] = str(value["metadata"]["gpu"])
                for w in value["workloads"]:
                    prefix = "len"+str(w["prompt_tokens"])
                    row.update({prefix+"_prefill_s": w["prefill_seconds"]["median"],
                        prefix+"_decode_tps": w["decode_tokens_per_second"],
                        prefix+"_peak_cuda_bytes": w["memory"]["cuda_peak_allocated"]})
            elif key == "clm":
                episodes[rid] = value["episodes"]
                row["clm_all_complete"] = value["all_complete"]
                for mode in ("clm", "append_only"):
                    scores = [e["rewards"]["reward"] for e in value["episodes"]
                              if e["mode"] == mode and e["status"] == "complete"]
                    row[mode+"_reward"] = statistics.mean(scores) if scores else None
        if (root/"failure.json").exists():
            row["recorded_failure"] = read_json(root/"failure.json")
        if len(set(identity.values())) > 1:
            raise ValueError("mixed artifact identities in run "+rid)
        if len(runtimes) > 1:
            raise ValueError("mixed inference precision policies in run "+rid)
        row["inference_runtime"] = next(iter(runtimes), None)
        row["missing"] = ";".join(missing)
        if not missing and row.get("training_status") == "complete" and row.get("clm_all_complete"):
            row["status"] = "complete"
        rows.append(row)
    return rows, episodes


def paired_summary(rows, episodes):
    """Hierarchical paired bootstrap: training seeds and evaluation-seed clusters.

    All levels generated with the same evaluation seed are resampled together.
    Tokens/turns are never treated as independent replications.
    """
    groups = {}
    for row in rows:
        if row["status"] != "complete" or row["method"] != "constrained":
            continue
        dense = next((r for r in rows if r["method"] == "dense_matched" and r["seed"] == row["seed"] and r["status"] == "complete"), None)
        if dense is None:
            continue
        if dense.get("inference_runtime") != row.get("inference_runtime"):
            raise ValueError("paired comparisons require the same inference precision policy")
        baseline = {(e["episode"], e["mode"]): e for e in episodes[dense["run_id"]]}
        for e in episodes[row["run_id"]]:
            b = baseline.get((e["episode"], e["mode"]))
            if b is None or b["status"] != "complete" or e["status"] != "complete":
                continue
            seed = re.search(r"seed(\d+)", e["episode"])
            if seed is None:
                raise ValueError("episode name lacks evaluation seed")
            family = e["episode"].split("-", 2)[0]  # unique families: kv, log, needle, sudoku
            key = (row["selected_keep_requested"], e["mode"], family)
            groups.setdefault(key, {}).setdefault(row["seed"], {}).setdefault(int(seed[1]), []).append(
                e["rewards"]["reward"]-b["rewards"]["reward"])
    result, rng = [], np.random.default_rng(42)
    for (keep, mode, family), values in groups.items():
        seeds = sorted(values)
        eval_seeds = sorted(set.intersection(*(set(values[s]) for s in seeds)))
        matrix = np.array([[np.mean(values[s][e]) for e in eval_seeds] for s in seeds])
        boot = [matrix[rng.integers(len(seeds), size=len(seeds))][:, rng.integers(len(eval_seeds), size=len(eval_seeds))].mean()
                for _ in range(2000)]
        result.append({"keep": keep, "mode": mode, "family": family, "mean_paired_reward_difference": float(matrix.mean()),
                       "bootstrap_95_percent": list(map(float, np.quantile(boot, [.025, .975]))),
                       "training_seeds": len(seeds), "evaluation_seed_clusters": len(eval_seeds),
                       "caution": "small-sample exploratory interval; not a publication-level power analysis"})
    return result


def gate(spec, rows, episodes):
    thresholds = spec["config"]["provisional_thresholds"]
    candidates = []
    baseline = spec.get("stage") == "baseline"
    if baseline:
        for row in rows:
            if row["status"] != "complete" or row["method"] != "original":
                continue
            scores = [e["rewards"]["reward"] for e in episodes[row["run_id"]] if e["mode"] == "clm"]
            candidates.append({"run_id": row["run_id"], "passed": bool(scores) and
                               all(s >= thresholds["min_dense_clm_reward"] for s in scores)})
    for row in rows:
        if row["status"] != "complete" or row["method"] != "constrained":
            continue
        dense = next((r for r in rows if r["method"] == "dense_matched" and r["seed"] == row["seed"] and r["status"] == "complete"), None)
        original = next((r for r in rows if r["method"] == "original" and r["status"] == "complete"), None)
        if not dense or not original:
            continue
        checks = {"smaller": row["whole_keep"] <= 1-thresholds["min_whole_model_reduction"],
                  "ppl_vs_matched": math.exp(row["nll"]-dense["nll"]) <= thresholds["max_perplexity_ratio"],
                  "ppl_vs_original": math.exp(row["nll"]-original["nll"]) <= thresholds["max_perplexity_ratio"],
                  "reload": row["reload_error"] <= 1e-4}
        for base in (dense, original):
            scores = {e["episode"]: e["rewards"]["reward"] for e in episodes[base["run_id"]] if e["mode"] == "clm"}
            for e in episodes[row["run_id"]]:
                if e["mode"] == "clm":
                    reference = scores[e["episode"]]
                    checks[base["method"]+"/"+e["episode"]] = (
                        reference >= thresholds["min_dense_clm_reward"] and
                        e["rewards"]["reward"] >= reference-thresholds["max_clm_absolute_drop"])
        candidates.append({"run_id": row["run_id"], "checks": checks, "passed": all(checks.values())})
    runtime_ok = bool(rows) and all(r.get("inference_runtime") == INFERENCE_RUNTIME for r in rows)
    return {"passed": runtime_ok and bool(candidates) and all(r["status"] == "complete" for r in rows) and any(c["passed"] for c in candidates),
            "inference_runtime_verified": runtime_ok,
            "gate_type": "baseline" if baseline else "pilot",
            "protocol_contract": protocol_contract(spec["config"], baseline=baseline),
            "thresholds_provisional": True, "thresholds": thresholds, "candidates": candidates,
            "model_id": spec["config"]["model"]["id"], "selection_kind": spec["config"]["selection"]["kind"],
            "pilot_commit": spec["commit"], "manifest_hash": digest(spec),
            "scope": "pilot-to-main screening only; no speedup required/claimed; failures remain in summary"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    spec = read_json(args.manifest)
    rows, episodes = collect(spec)
    out = Path(spec["campaign"])/"summary"
    out.mkdir(exist_ok=True)
    write_json(out/"runs.json", rows)
    fields = sorted({k for r in rows for k in r})
    with (out/"runs.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out/"paired_clm.json", paired_summary(rows, episodes) if spec["stage"] != "ablations" else
               {"note": "one-seed focused controls; compare each run ID without pooling schedules or layer policies"})
    write_json(out/"gate.json", gate(spec, rows, episodes))
    if any(r["status"] != "complete" for r in rows):
        raise SystemExit("Incomplete/failed runs included in summary; campaign is NOT successful.")


if __name__ == "__main__":
    main()
