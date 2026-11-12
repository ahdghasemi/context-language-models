"""Bounded prompt-format experiment reusing a completed screen's dense export."""
from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from .answer_format import FORMAT_ID, TOOL_FORMAT_ID, protocol_identity, protocol_text
from .provenance import verify_snapshot, INFERENCE_RUNTIME
from .screen_contract import screen_report, export_verified


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_source(manifest, *, verify_weights=False):
    source = read(manifest)
    root = Path(source["campaign"])
    if (source["stage"] != "backbone_screen" or source.get("diagnostic_protocol") is not None
            or read(root/"config.json") != source["config"]):
        raise ValueError("requires an original-protocol backbone screen")
    cfg = source["config"]
    if (len(cfg["screen"]["task_names"]) != 1 or cfg["contextbench"]["modes"] != ["clm", "append_only"]
            or not all(n.startswith("kv-store-") for n in cfg["screen"]["task_names"])):
        raise ValueError("format diagnostic requires one KV task in both modes")
    lock_path = Path(source["contextbench_lock"])
    if sha(lock_path) != source["contextbench_lock_sha256"]:
        raise ValueError("source task lock changed")
    lock = read(lock_path)
    audit = read(root/"export-audit.json")
    summary = read(root/"contextbench/summary.json")
    computed = screen_report(cfg, lock, audit, summary)
    original = read(root/"screen-report.json")
    if (not computed["complete"] or original.get("manifest_sha256") != sha(manifest)
            or any(original.get(k) != computed[k] for k in ("complete", "passed", "episodes", "model", "export_verified"))):
        raise ValueError("source screen is incomplete or inconsistent; a valid zero score is allowed")
    artifact = read(root/"export/compression.json")
    if (not export_verified(audit, cfg["model"]) or artifact.get("ranks") != {}
            or artifact.get("whole_keep") != 1 or artifact.get("dtype") != "bfloat16"
            or artifact.get("inference_runtime") != INFERENCE_RUNTIME
            or artifact.get("weight_sha256") != audit["weight_sha256"]):
        raise ValueError("requires a verified uncompressed BF16 export")
    if verify_weights and sha(root/"export/model.safetensors") != artifact["weight_sha256"]:
        raise ValueError("export weights changed")
    return source, lock, audit, summary


TOOL_SCREEN_TASKS = ["kv-store-R1000-seed1101", "kv-store-R130-seed1102"]


def validate_reference(manifest, identifier=FORMAT_ID, *, verify_weights=False):
    """Original screen for answer-only; completed validation for tool guidance.

    A valid negative source is allowed. Never silently turn a diagnostic into
    a baseline gate or change task locks to accommodate a prompt intervention.
    """
    protocol_identity(identifier)
    if identifier == FORMAT_ID:
        source, lock, audit, baseline = validate_source(manifest, verify_weights=verify_weights)
        return source, lock, audit, baseline, source["config"]
    if identifier != TOOL_FORMAT_ID:
        raise ValueError("unsupported screen protocol")
    from .protocol_validation import verify_inputs as verify_validation, validation_report
    parent = read(manifest)
    root = Path(parent["campaign"])
    if (parent.get("stage") != "protocol_validation" or
            read(root/"config.json") != parent["config"] or
            (root/"reminder.txt").read_text() != protocol_text(FORMAT_ID)):
        raise ValueError("tool-format screen requires the completed answer-format validation")
    dev, source, lock, audit = verify_validation(parent, verify_weights=verify_weights)
    baseline = read(root/"contextbench/summary.json")
    parent["manifest_sha256"] = sha(manifest)
    computed = validation_report(parent, dev, lock, audit, baseline, read(root/"memory-probe.json"))
    saved = read(root/"validation-report.json")
    checks = read(root/"checks.json")
    if (not computed["complete"] or saved.get("manifest_sha256") != sha(manifest) or
            checks.get("passed") is not True or checks.get("manifest_sha256") != sha(manifest) or
            any(saved.get(k) != computed[k] for k in computed)):
        raise ValueError("validation source report is incomplete or inconsistent")
    for ep in baseline["episodes"]:
        if read(root/"contextbench"/(ep["mode"]+"--"+ep["episode"]+".json")) != ep:
            raise ValueError("source validation episode differs from summary")
    cfg = copy.deepcopy(parent["config"])
    if not set(TOOL_SCREEN_TASKS) <= set(cfg["screen"]["task_names"]):
        raise ValueError("source lacks required diagnostic tasks")
    cfg["screen"]["task_names"] = list(TOOL_SCREEN_TASKS)
    return source, lock, audit, baseline, cfg


def evidence_files(manifest, source, identifier=FORMAT_ID):
    if identifier == TOOL_FORMAT_ID:
        parent = read(manifest)
        root = Path(parent["campaign"])
        paths = [Path(manifest)] + [root/name for name in ("config.json", "reminder.txt", "checks.json",
            "validation-report.json", "memory-probe.json", "contextbench/summary.json")]
        paths += [root/"contextbench"/(ep["mode"]+"--"+ep["episode"]+".json")
                  for ep in read(root/"contextbench/summary.json")["episodes"]]
        return {**parent["source_files"], **{str(p.resolve()): sha(p) for p in paths}}
    if identifier != FORMAT_ID:
        raise ValueError("unknown source evidence protocol")
    root = Path(source["campaign"])
    paths = [Path(manifest), Path(source["contextbench_lock"])]
    paths += [root/name for name in ("config.json", "export-audit.json", "screen-report.json", "contextbench/summary.json")]
    paths += sorted(p for p in (root/"export").rglob("*") if p.is_file() and p.name != "model.safetensors")
    paths += [root/"contextbench"/(m+"--"+t+".json") for m in source["config"]["contextbench"]["modes"]
              for t in source["config"]["screen"]["task_names"]]
    return {str(p.resolve()): sha(p) for p in paths}


def verify_inputs(spec, *, verify_weights=False):
    identifier = spec["diagnostic_protocol"]["id"]
    source, lock, audit, summary, cfg = validate_reference(spec["source_manifest"], identifier, verify_weights=verify_weights)
    if (cfg != spec["config"] or evidence_files(spec["source_manifest"], source, identifier) != spec["source_files"]
            or spec.get("diagnostic_protocol") != protocol_identity(identifier)):
        raise ValueError("source evidence or declared protocol changed after submission")
    return source, lock, audit, summary


def compare_report(config, lock, audit, baseline, variant, *, diagnostic_protocol=FORMAT_ID):
    report = screen_report(config, lock, audit, variant, diagnostic_protocol=diagnostic_protocol)
    controls = {(e["mode"], e["episode"]): e for e in baseline["episodes"]}
    report["comparison"] = [{"mode": e["mode"], "episode": e["episode"],
        "original_reward": controls[e["mode"], e["episode"]]["rewards"]["reward"],
        "reminder_reward": e["rewards"]["reward"],
        "original_answered": controls[e["mode"], e["episode"]]["rewards"].get("n_answered"),
        "reminder_answered": e["rewards"].get("n_answered")}
        for e in report["episodes"]] if report["complete"] else []
    report["scope"] = ("one development task; historical original-prompt control and one formatting-reminder run per mode; "
                       "no original-score revision, compression authorization, statistical generalization or speedup claim")
    if diagnostic_protocol == TOOL_FORMAT_ID:
        # The control here already includes answer-format-v1, not the original prompt.
        for row in report["comparison"]:
            row["answer_format_control_reward"] = row.pop("original_reward")
            row["tool_guidance_reward"] = row.pop("reminder_reward")
            row["answer_format_control_answered"] = row.pop("original_answered")
            row["tool_guidance_answered"] = row.pop("reminder_answered")
        report.update(gate_type="tool_call_screen", control_protocol=protocol_identity(FORMAT_ID),
            scope="two previously failed tasks in both modes; initial tool-call guidance added to the fixed answer reminder; "
                  "historical controls; unchanged parser, execution, decoding and grading; "
                  "development diagnostic only, no compression authorization or held-out-quality claim")
    return report


def main():
    from .io import write_json, metadata
    manifest, stage = Path(sys.argv[1]), sys.argv[2]
    spec = read(manifest)
    identifier = spec["diagnostic_protocol"]["id"]
    campaign, snapshot = Path(spec["campaign"]), Path(spec["snapshot"])
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("protocol stages require a Slurm allocation")
    verify_snapshot(snapshot, spec["snapshot_hash"])
    if (spec["stage"] != "protocol_screen" or read(campaign/"config.json") != spec["config"]
            or (campaign/"reminder.txt").read_text() != protocol_text(identifier)):
        raise ValueError("protocol campaign configuration differs")
    source, lock, audit, baseline = verify_inputs(spec, verify_weights=stage != "protocol_report")
    if stage == "protocol_check":
        from .contextbench import verify_lock
        verify_lock(lock, spec["config"]["contextbench"])
        subprocess.run([sys.executable, "-m", "pytest", "-q", "compression/tests", "clm/tests/test_context_string.py",
            "--junitxml="+str(campaign/"correctness.xml"), "-o", "cache_dir="+str(campaign/"pytest-cache")], check=True)
        env = dict(os.environ, PYTHONPATH=lock["source"]+os.pathsep+str(snapshot/"clm"))
        # Exercise real installed Harbor/ContextBench imports before allocating the GPU stage.
        subprocess.run([sys.executable, "-c", "from tiny_clm.contextbench_protocol import FormatReminderClm, FormatReminderAppendOnly, ToolFormatReminderClm, ToolFormatReminderAppendOnly; "
            "from clm_contextbench import ClmStream; from contextbench.harbor.agents.base import ContextBenchAgent; "
            "assert issubclass(FormatReminderClm, ClmStream); assert issubclass(FormatReminderAppendOnly, ContextBenchAgent); "
            "assert issubclass(ToolFormatReminderClm, ClmStream); assert issubclass(ToolFormatReminderAppendOnly, ContextBenchAgent); "
            "print('FORMAT_ADAPTER_IMPORTS_PASSED')"], env=env, check=True)
        write_json(campaign/"checks.json", {"passed": True, "diagnostic_protocol": spec["diagnostic_protocol"], "metadata": metadata()})
    elif stage == "protocol_contextbench":
        import torch
        from .contextbench import run
        if not read(campaign/"checks.json").get("passed"):
            raise ValueError("protocol checks did not pass")
        if (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()
                or torch.cuda.get_device_properties(0).total_memory < 32*2**30):
            raise RuntimeError("requires one allocated BF16 GPU with at least 32 GiB")
        if (campaign/"contextbench").exists():
            raise FileExistsError("preserve prior or partial results; use a new campaign")
        run(Path(source["campaign"])/"export", spec["config"], source["contextbench_lock"], campaign/"contextbench",
            task_names=spec["config"]["screen"]["task_names"],
            server_timeout=spec["config"]["screen"]["server_start_timeout"], diagnostic_protocol=identifier)
    elif stage == "protocol_report":
        path = campaign/"contextbench/summary.json"
        report = compare_report(spec["config"], lock, audit, baseline, read(path) if path.exists() else None,
            diagnostic_protocol=identifier)
        checks = campaign/"checks.json"
        if not checks.exists() or not read(checks).get("passed"):
            report["errors"].append("prerequisite correctness checks did not pass")
            report.update(complete=False, passed=False)
        report.update(commit=spec["commit"], manifest_sha256=sha(manifest),
            source_manifest_sha256=spec["source_files"][str(Path(spec["source_manifest"]).resolve())], metadata=metadata())
        write_json(campaign/"protocol-report.json", report)
        print(json.dumps(report, indent=2))
        if not report["complete"]:
            raise RuntimeError("incomplete protocol experiment; inspect protocol-report.json and stage logs")
    else:
        raise ValueError("unknown protocol stage")


if __name__ == "__main__":
    main()
