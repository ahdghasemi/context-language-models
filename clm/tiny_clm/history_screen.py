"""Fresh paired CLM episodes after a verified inference-only history replay."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from .answer_format import TOOL_FORMAT_ID
from .history_audit import CONTRACT, digest, prepare_source, valid_bash
from .history_format import CONTROL, NATIVE, identity, audit_log
from .protocol_screen import read, sha, verify_inputs as verify_tool_source
from .provenance import verify_snapshot, INFERENCE_RUNTIME
from .screen_contract import screen_report

CONTRACT_SCREEN = "paired-history-episodes-v1"
ARMS = (("control", CONTROL), ("native_history", NATIVE))


def validate_replay(manifest, *, verify_weights=False):
    manifest = Path(manifest).resolve()
    spec = read(manifest)
    root = Path(spec["campaign"])
    source, bundle = prepare_source(spec["source_manifest"], verify_weights=verify_weights)
    report = read(root/"history-audit.json")
    if (spec.get("stage") != "history_audit" or spec.get("contract") != CONTRACT or
            spec["config"] != source["config"] or read(root/"config.json") != spec["config"] or
            bundle != read(root/"replay-inputs.json") or digest(bundle) != spec["inputs_sha256"] or
            report.get("manifest_sha256") != sha(manifest) or report.get("contract") != CONTRACT or
            report.get("source_inputs_sha256") != spec["inputs_sha256"] or
            report.get("historical_weight_sha256") != bundle["weight_sha256"] or
            report.get("metadata", {}).get("commit") != spec["commit"] or
            report.get("inference_runtime") != INFERENCE_RUNTIME or report.get("error") is not None or
            report.get("diagnostic_complete") is not True or report.get("can_advance_to_compression") is not False):
        raise ValueError("history replay is incomplete, incompatible or changed")
    observations = report.get("observations", [])
    if [o.get("variant") for o in observations] != ["original", "native_history", "native_history", "original"]:
        raise ValueError("expected the four ordered history replay completions")
    for row in observations:
        request = {**copy.deepcopy(bundle["request"]), "messages": bundle[row["variant"]]}
        if row.get("request_sha256") != digest(request) or row.get("valid_bash_call") is not valid_bash(row["response"]):
            raise ValueError("replay request/response audit differs")
    original, native = observations[0]["response"], observations[1]["response"]
    if (original != observations[3]["response"] or native != observations[2]["response"] or
            original["usage"]["prompt_tokens"] != bundle["historical_request"]["prompt_tokens"] or
            original["message"].get("tool_calls") or original["finish_reason"] != "stop" or not valid_bash(native) or
            report.get("format_hypothesis_supported_on_this_prompt") is not True):
        raise ValueError("replay does not support the declared bounded history experiment")
    paths = [manifest, root/"config.json", root/"replay-inputs.json", root/"history-audit.json"]
    files = {**bundle["source_files"], **{str(p.resolve()): sha(p) for p in paths}}
    return source, bundle, files


def verify_inputs(spec, *, verify_weights=False):
    source, bundle, files = validate_replay(spec["replay_manifest"], verify_weights=verify_weights)
    if (spec.get("stage") != "history_screen" or spec.get("contract") != CONTRACT_SCREEN or
            files != spec["source_files"] or spec["config"] != source["config"]):
        raise ValueError("history screen source evidence or configuration changed")
    original, lock, audit, _ = verify_tool_source(source)
    return source, original, lock, audit


def paired_report(config, lock, audit, summaries):
    reports = {name: screen_report(config, lock, audit, summaries.get(name),
        diagnostic_protocol=TOOL_FORMAT_ID, modes=["clm"], history_protocol=identifier)
        for name, identifier in ARMS}
    errors = [name+": "+error for name, report in reports.items() for error in report["errors"]]
    for name, identifier in ARMS:
        for ep in (summaries.get(name) or {}).get("episodes", []):
            application = ep.get("history_application") or {}
            if (application.get("history_protocol") != identity(identifier) or
                    not isinstance(application.get("requests"), int) or application["requests"] < 1 or
                    not isinstance(application.get("changed_historical_calls"), int) or
                    application["changed_historical_calls"] < 0 or
                    (identifier == CONTROL and application["changed_historical_calls"] != 0)):
                errors.append(name+": missing or invalid history application: "+str(ep.get("episode")))
    complete = not errors
    comparisons = []
    if complete:
        by_arm = {name: {e["episode"]: e for e in summaries[name]["episodes"]} for name, _ in ARMS}
        for task in config["screen"]["task_names"]:
            row = {"episode": task, "mode": "clm"}
            for name, _ in ARMS:
                ep = by_arm[name][task]
                row[name] = {"reward": ep["rewards"]["reward"], "answered": ep["rewards"].get("n_answered"),
                    **{k: ep.get(k) for k in ("tool_calls", "model_calls", "prompt_tokens", "completion_tokens", "elapsed_seconds")},
                    "history_application": ep["history_application"]}
            comparisons.append(row)
    applied = complete and all(row["native_history"]["history_application"]["changed_historical_calls"] > 0
                               for row in comparisons)
    return {"gate_type": "history_screen", "contract": CONTRACT_SCREEN, "complete": complete,
        "passed": bool(complete and applied and reports["native_history"]["passed"]),
        "native_history_applied_all_tasks": bool(applied), "can_advance_to_compression": False,
        "errors": errors, "comparison": comparisons, "arms": reports,
        "min_clm_reward": config["screen"]["min_clm_reward"], "threshold_provisional": True,
        "scope": "two previously inspected development tasks, fresh control and native-history CLM episodes; "
                 "one greedy run per arm/task, original execution and grading; no held-out validation, "
                 "compression authorization, statistical significance or speedup claim"}


def main():
    from .io import write_json, metadata
    manifest, stage = Path(sys.argv[1]), sys.argv[2]
    spec = read(manifest)
    root, snapshot = Path(spec["campaign"]), Path(spec["snapshot"])
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("history screen requires a Slurm allocation")
    verify_snapshot(snapshot, spec["snapshot_hash"])
    if read(root/"config.json") != spec["config"]:
        raise ValueError("campaign configuration changed")
    _, original, lock, audit = verify_inputs(spec, verify_weights=stage != "history_report")
    if stage == "history_check":
        from .contextbench import verify_lock
        verify_lock(lock, spec["config"]["contextbench"])
        subprocess.run([sys.executable, "-m", "pytest", "-q", "compression/tests", "clm/tests/test_context_string.py",
            "--junitxml="+str(root/"correctness.xml"), "-o", "cache_dir="+str(root/"pytest-cache")], check=True)
        env = dict(os.environ, PYTHONPATH=lock["source"]+os.pathsep+str(snapshot/"clm"))
        subprocess.run([sys.executable, "-c", "from tiny_clm.contextbench_protocol import HistoryControlClm, NativeHistoryClm; "
            "from clm_contextbench import ClmStream; "
            "assert issubclass(HistoryControlClm, ClmStream) and issubclass(NativeHistoryClm, ClmStream); "
            "print('HISTORY_ADAPTER_IMPORTS_PASSED')"], env=env, check=True)
        write_json(root/"checks.json", {"passed": True, "manifest_sha256": sha(manifest), "metadata": metadata()})
    elif stage == "history_contextbench":
        import torch
        from .contextbench import run
        checks = read(root/"checks.json")
        if checks.get("passed") is not True or checks.get("manifest_sha256") != sha(manifest):
            raise ValueError("history checks did not pass for this manifest")
        if (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported() or
                torch.cuda.get_device_properties(0).total_memory < 32*2**30):
            raise RuntimeError("requires one allocated BF16 GPU with at least 32 GiB")
        if any((root/name).exists() for name, _ in ARMS):
            raise FileExistsError("preserve partial episodes; choose a new campaign")
        # Keep controls even if one arm has an infrastructure failure. A negative
        # graded reward is not an exception and never prevents the other arm.
        failures = []
        for name, identifier in ARMS:
            try:
                run(Path(original["campaign"])/"export", spec["config"], original["contextbench_lock"], root/name,
                    task_names=spec["config"]["screen"]["task_names"], modes=["clm"],
                    server_timeout=spec["config"]["screen"]["server_start_timeout"],
                    diagnostic_protocol=TOOL_FORMAT_ID, history_protocol=identifier)
            except Exception as exc:
                failures.append({"arm": name, "error": f"{type(exc).__name__}: {exc}"})
        write_json(root/"execution.json", {"complete": not failures, "failures": failures, "metadata": metadata()})
        if failures:
            raise RuntimeError("one or more history arms failed; inspect execution.json")
    elif stage == "history_report":
        summaries, evidence_errors = {}, []
        for name, identifier in ARMS:
            path = root/name/"summary.json"
            summaries[name] = read(path) if path.exists() else None
            for ep in (summaries[name] or {}).get("episodes", []):
                episode = ep["mode"]+"--"+ep["episode"]
                try:
                    if read(root/name/(episode+".json")) != ep or audit_log(
                            root/name/"trials"/episode/"agent/history_format.jsonl", identifier) != ep.get("history_application"):
                        raise ValueError("episode or history evidence differs from summary")
                except (ValueError, OSError) as exc:
                    evidence_errors.append(name+"/"+episode+": "+str(exc))
        report = paired_report(spec["config"], lock, audit, summaries)
        checks = read(root/"checks.json") if (root/"checks.json").exists() else {}
        execution = read(root/"execution.json") if (root/"execution.json").exists() else {}
        if checks.get("passed") is not True or checks.get("manifest_sha256") != sha(manifest):
            evidence_errors.append("prerequisite checks missing or incompatible")
        if execution.get("complete") is not True or execution.get("failures") != []:
            evidence_errors.append("episode execution incomplete; inspect execution.json")
        if evidence_errors:
            report["errors"].extend(evidence_errors)
            report.update(complete=False, passed=False)
        report.update(commit=spec["commit"], manifest_sha256=sha(manifest), metadata=metadata())
        write_json(root/"history-screen.json", report)
        print(json.dumps(report, indent=2))
        if not report["complete"]:
            raise RuntimeError("incomplete history experiment; inspect history-screen.json")
    else:
        raise ValueError("unknown history screen stage")


if __name__ == "__main__":
    main()
