"""Prospectively fixed four-task behavior validation plus a separate memory probe."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from .answer_format import FORMAT_ID, FORMAT_TEXT, protocol_identity
from .protocol_screen import read, sha, verify_inputs as verify_development_inputs, compare_report
from .provenance import verify_snapshot
from .screen_contract import screen_report
from .training_memory import probe_config, estimates


def development(manifest, *, verify_weights=False):
    spec = read(manifest)
    if spec.get("stage") != "protocol_screen" or spec.get("diagnostic_protocol") != protocol_identity(FORMAT_ID):
        raise ValueError("requires the completed format-development manifest")
    root = Path(spec["campaign"])
    if read(root/"config.json") != spec["config"] or (root/"reminder.txt").read_text() != FORMAT_TEXT:
        raise ValueError("development configuration changed")
    source, lock, audit, baseline = verify_development_inputs(spec, verify_weights=verify_weights)
    variant = read(root/"contextbench/summary.json")
    computed = compare_report(spec["config"], lock, audit, baseline, variant)
    saved = read(root/"protocol-report.json")
    if (not computed["passed"] or saved.get("manifest_sha256") != sha(manifest) or
            not read(root/"checks.json").get("passed") or
            any(saved.get(k) != computed[k] for k in computed)):
        raise ValueError("requires a passing, consistent development protocol report")
    return spec, source, lock, audit


def expanded_config(spec, lock):
    cfg = copy.deepcopy(spec["config"])
    names = [Path(t["path"]).name for t in lock["tasks"]]
    expected = {f"kv-store-{level}-seed{seed}" for level in ("R130", "R1000") for seed in (1101, 1102)}
    if len(names) != 4 or set(names) != expected:
        raise ValueError("validation requires the existing four-task R130/R1000 lock, seeds 1101/1102")
    cfg["screen"]["task_names"] = names
    return cfg


def evidence_files(manifest, spec):
    root = Path(spec["campaign"])
    files = dict(spec["source_files"])
    paths = [Path(manifest)] + [root/n for n in ("config.json", "reminder.txt", "checks.json",
        "protocol-report.json", "contextbench/summary.json")]
    paths += [root/"contextbench"/(mode+"--"+task+".json")
        for mode in spec["config"]["contextbench"]["modes"] for task in spec["config"]["screen"]["task_names"]]
    files.update({str(p.resolve()): sha(p) for p in paths})
    return files


def verify_inputs(spec, *, verify_weights=False):
    dev, source, lock, audit = development(spec["development_manifest"], verify_weights=verify_weights)
    if (expanded_config(dev, lock) != spec["config"] or
            spec["source_files"] != evidence_files(spec["development_manifest"], dev) or
            spec["diagnostic_protocol"] != protocol_identity(FORMAT_ID) or
            spec["memory_config"] != probe_config(spec["config"])):
        raise ValueError("validation inputs changed after submission")
    return dev, source, lock, audit


def validation_report(spec, dev, lock, audit, summary, memory):
    report = screen_report(spec["config"], lock, audit, summary, diagnostic_protocol=FORMAT_ID)
    report.update(gate_type="protocol_validation", dense_behavior_passed=report["passed"],
                  episodes_complete=report["complete"], analytical_memory=estimates(spec["config"]))
    dev_tasks = set(dev["config"]["screen"]["task_names"])
    report["cohorts"] = {}
    for mode in spec["config"]["contextbench"]["modes"]:
        for cohort in ("development_repeat", "additional_validation"):
            selected = [e for e in report["episodes"] if e["mode"] == mode and
                ((e["episode"] in dev_tasks) == (cohort == "development_repeat"))]
            scores = [e["rewards"]["reward"] for e in selected] if report["episodes_complete"] else []
            report["cohorts"][mode+"/"+cohort] = {"tasks": [e["episode"] for e in selected],
                "rewards": scores, "mean": sum(scores)/len(scores) if scores else None,
                "minimum": min(scores) if scores else None}
    estimate = estimates(spec["config"])
    metrics = (memory or {}).get("memory", {})
    reserved, allocated = metrics.get("cuda_peak_reserved"), metrics.get("cuda_peak_allocated")
    total = (memory or {}).get("device_total_bytes")
    measurements_valid = (all(type(n) is int and n > 0 for n in (total, allocated, reserved)) and
        allocated <= reserved <= total)
    valid_memory = bool(memory and memory.get("model") == spec["config"]["model"] and
        memory.get("training_config") == spec["memory_config"] and memory.get("sequence_length") == 512 and
        memory.get("weight_sha256") == audit["weight_sha256"] and
        memory.get("manifest_sha256") == spec["manifest_sha256"] and measurements_valid and
        memory.get("trainer_status", {}).get("total_steps") == 2 and
        memory.get("trainer_status", {}).get("training_tokens") == 4088 and
        memory.get("budget", {}).get("original_parameters") == estimate["parameters"] and
        memory.get("budget", {}).get("deployment_parameters") == estimate["predicted_deployment_parameters"] and
        memory.get("reserved_headroom_bytes") == total-reserved)
    report["memory_probe"] = memory
    report["memory_probe_passed"] = bool(valid_memory and memory.get("passed") and memory.get("probe_complete") and
        memory["reserved_headroom_bytes"] >= estimate["minimum_measured_headroom_bytes"])
    if not valid_memory or not memory.get("probe_complete"):
        report["errors"].append("missing, failed, or incompatible memory probe; inspect memory-probe.json")
    report["complete"] = report["episodes_complete"] and valid_memory and bool(memory.get("probe_complete"))
    report["passed"] = report["dense_behavior_passed"] and report["memory_probe_passed"]
    report["scope"] = ("dense behavior under fixed kv-answer-format-v1, with development task reported separately; "
        "two seeds and no final test; synthetic-token memory probe only; no compressed model, general-LM quality or speedup result")
    report["remaining_before_quality_pilot"] = ["dense language-model validation and inference profiling",
        "matched-data training design using measured memory; preserve this prompt protocol for every method",
        "append-only command-generation control remains unresolved if tool calls are still zero"]
    return report


def main():
    from .io import write_json, metadata
    manifest, stage = Path(sys.argv[1]), sys.argv[2]
    spec = read(manifest)
    campaign, snapshot = Path(spec["campaign"]), Path(spec["snapshot"])
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("validation stages require a Slurm allocation")
    verify_snapshot(snapshot, spec["snapshot_hash"])
    if (spec.get("stage") != "protocol_validation" or read(campaign/"config.json") != spec["config"] or
            (campaign/"reminder.txt").read_text() != FORMAT_TEXT):
        raise ValueError("validation configuration changed")
    dev, source, lock, audit = verify_inputs(spec, verify_weights=stage != "validation_report")
    spec["manifest_sha256"] = sha(manifest)
    if stage == "validation_check":
        from .contextbench import verify_lock
        verify_lock(lock, spec["config"]["contextbench"])
        subprocess.run([sys.executable, "-m", "pytest", "-q", "compression/tests", "clm/tests/test_context_string.py",
            "--junitxml="+str(campaign/"correctness.xml"), "-o", "cache_dir="+str(campaign/"pytest-cache")], check=True)
        env = dict(os.environ, PYTHONPATH=lock["source"]+os.pathsep+str(snapshot/"clm"))
        subprocess.run([sys.executable, "-c", "from tiny_clm.contextbench_protocol import FormatReminderClm, FormatReminderAppendOnly; "
            "print('FORMAT_ADAPTER_IMPORTS_PASSED')"], env=env, check=True)
        write_json(campaign/"checks.json", {"passed": True, "manifest_sha256": sha(manifest), "metadata": metadata()})
    elif stage in {"validation_contextbench", "validation_memory"}:
        checks = read(campaign/"checks.json")
        if checks.get("passed") is not True or checks.get("manifest_sha256") != sha(manifest):
            raise ValueError("validation checks did not pass")
        artifact = Path(source["campaign"])/"export"
        if stage == "validation_contextbench":
            import torch
            from .contextbench import run
            if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
                raise RuntimeError("requires exactly one allocated BF16 GPU")
            if (campaign/"contextbench").exists():
                raise FileExistsError("preserve partial results; use a new campaign")
            run(artifact, spec["config"], source["contextbench_lock"], campaign/"contextbench",
                task_names=spec["config"]["screen"]["task_names"],
                server_timeout=spec["config"]["screen"]["server_start_timeout"], diagnostic_protocol=FORMAT_ID)
        else:
            from .training_memory import probe
            path = campaign/"memory-probe.json"
            result = probe(artifact, spec["config"], path)
            result["manifest_sha256"] = sha(manifest)
            write_json(path, result)
            if not result["passed"]:
                raise RuntimeError("memory probe failed; see memory-probe.json; do not launch a training grid")
    elif stage == "validation_report":
        def optional(name):
            path = campaign/name
            return read(path) if path.exists() else None
        report = validation_report(spec, dev, lock, audit, optional("contextbench/summary.json"), optional("memory-probe.json"))
        checks = optional("checks.json")
        if not checks or not checks.get("passed") or checks.get("manifest_sha256") != sha(manifest):
            report["errors"].append("prerequisite checks missing or incompatible")
            report.update(complete=False, passed=False)
        report.update(commit=spec["commit"], manifest_sha256=sha(manifest), metadata=metadata())
        write_json(campaign/"validation-report.json", report)
        print(json.dumps(report, indent=2))
        if not report["complete"]:
            raise RuntimeError("incomplete validation; inspect validation-report.json and stage logs")
    else:
        raise ValueError("unknown validation stage")


if __name__ == "__main__":
    main()
