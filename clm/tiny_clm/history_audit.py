"""Inference-only history-format ablation; never execute generated commands."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from clm_harness.utils.tool_schemas import BASH_TOOL
from .answer_format import TOOL_FORMAT_ID, protocol_identity, protocol_text
from .protocol_screen import read, sha, verify_inputs as verify_source, compare_report
from .provenance import verify_snapshot, INFERENCE_RUNTIME


EPISODE = "clm--kv-store-R1000-seed1101"
CONTRACT = "verified-history-envelope-replay-v1"
GET = re.compile(r"^=== get (\d+) \(\d+ op\(s\) remaining after this\) ===\s*\nGET (K\d+)\s*$", re.MULTILINE)
PLAIN_CALL = re.compile(r"(?m)^bash (\{[^\n]*\})[ \t]*\Z")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def call_key(function):
    args = function.get("arguments")
    if isinstance(args, str):
        args = json.loads(args)
    if function.get("name") != "bash" or not isinstance(args, dict) or not isinstance(args.get("command"), str):
        raise ValueError("expected an observed bash call with string command")
    return digest({"name": "bash", "arguments": args})


def native_history(messages, observed_calls):
    """Re-encode only historical commands matched to earlier structured calls.

    This changes assistant content text, not message roles, structured tool
    fields, command arguments, protected instructions, observations or GET.
    Unknown/mutated command strings fail closed. No text becomes an action.
    """
    result, changes = copy.deepcopy(messages), []
    for index, message in enumerate(result):
        if index < 2 or message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        def replace(match):
            args = json.loads(match[1])
            key = call_key({"name": "bash", "arguments": args})
            if key not in observed_calls:
                raise ValueError("historical command has no matching earlier structured call")
            changes.append({"message_index": index, "command_sha256": key,
                            "observed_source": observed_calls[key]})
            return "<tool_call>\n"+json.dumps({"name": "bash", "arguments": args}, ensure_ascii=False)+"\n</tool_call>"
        message["content"] = PLAIN_CALL.sub(replace, content)
    if not changes:
        raise ValueError("no verified historical plain calls to re-encode")
    return result, changes


def first_get_input(agent):
    """Select a tagged pre-call snapshot, with evidence only from earlier files."""
    known, files = {}, {}
    paths = sorted((agent/"context_snapshots").glob("turn-*.json"),
                   key=lambda p: int(p.stem.split("-")[1]))
    for path in paths:
        value = read(path)
        if not isinstance(value, dict) or value.get("kind") != "agent":
            continue  # Untagged duplicates/final snapshots are not LM inputs.
        if value.get("step") != int(path.stem.split("-")[1]):
            raise ValueError("snapshot step differs from filename")
        files[str(path.resolve())] = sha(path)
        messages = value["messages"]
        gets = [(i, GET.findall(m.get("content", ""))) for i, m in enumerate(messages)
                if m.get("role") == "user" and isinstance(m.get("content"), str)]
        gets = [(i, matches) for i, matches in gets if matches]
        if gets:
            index, matches = gets[0]
            if len(gets) != 1 or len(matches) != 1 or any(m.get("role") == "assistant" for m in messages[index+1:]):
                raise ValueError("first retained GET snapshot already contains a response or multiple GETs")
            native, changes = native_history(messages, known)
            return {"original": messages, "native_history": native, "changes": changes,
                    "query": {"qid": int(matches[0][0]), "key": matches[0][1]},
                    "snapshot": str(path.resolve()), "snapshot_step": value["step"],
                    "source_files": files}
        for message in messages:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    key = call_key(call["function"])
                    known.setdefault(key, str(path.resolve()))
    raise ValueError("no tagged pre-call snapshot containing the first GET")


def prepare_source(manifest, *, verify_weights=False):
    manifest = Path(manifest).resolve()
    source = read(manifest)
    root = Path(source["campaign"])
    if source.get("stage") != "protocol_screen" or source.get("diagnostic_protocol") != protocol_identity(TOOL_FORMAT_ID):
        raise ValueError("requires the completed native-tool-guidance screen")
    if (read(root/"config.json") != source["config"] or
            (root/"reminder.txt").read_text() != protocol_text(TOOL_FORMAT_ID)):
        raise ValueError("source configuration changed")
    original, lock, audit, baseline = verify_source(source, verify_weights=verify_weights)
    summary = read(root/"contextbench/summary.json")
    computed = compare_report(source["config"], lock, audit, baseline, summary, diagnostic_protocol=TOOL_FORMAT_ID)
    report = read(root/"protocol-report.json")
    checks = read(root/"checks.json")
    if (not computed["complete"] or report.get("manifest_sha256") != sha(manifest) or
            checks.get("passed") is not True or checks.get("diagnostic_protocol") != protocol_identity(TOOL_FORMAT_ID) or
            any(report.get(k) != computed[k] for k in computed)):
        raise ValueError("source screen is incomplete or inconsistent")
    record = read(root/"contextbench"/(EPISODE+".json"))
    if record not in summary["episodes"] or record.get("weight_sha256") != audit["weight_sha256"]:
        raise ValueError("source episode or weights differ from verified report")
    bundle = first_get_input(root/"contextbench/trials"/EPISODE/"agent")
    requests_path = root/"contextbench"/(EPISODE+".requests.jsonl")
    rows = [json.loads(line) for line in requests_path.read_text().splitlines() if line.strip()]
    step = bundle["snapshot_step"]
    if (len(rows) != record["model_calls"] or not 0 < step < len(rows) or
            any(row.get("weight_sha256") != audit["weight_sha256"] for row in rows) or
            rows[step].get("tool_calls") != 0 or rows[step].get("finish_reason") != "stop" or
            rows[step-1].get("tool_calls", 0) < 1):
        raise ValueError("saved first GET does not match the observed parsed-call to non-call transition")
    cb = source["config"]["contextbench"]
    if cb["temperature"] != 0 or cb["max_tokens"] != 1024 or cb["enable_thinking"] is not False:
        raise ValueError("audit supports the observed deterministic non-thinking 1024-token protocol")
    names = [Path(task["path"]).name for task in lock["tasks"]]
    seed = 10000+names.index(EPISODE.split("--", 1)[1])
    request = {"model": "tiny-clm", "tools": [BASH_TOOL], "max_tokens": cb["max_tokens"],
               "temperature": cb["temperature"], "top_p": cb["top_p"],
               "chat_template_kwargs": {"enable_thinking": cb["enable_thinking"]}}
    bundle.update(contract=CONTRACT, request=request, seed=seed, historical_request=rows[step],
                  weight_sha256=audit["weight_sha256"], artifact=str(Path(original["campaign"])/"export"))
    paths = [manifest, root/"config.json", root/"reminder.txt", root/"checks.json", root/"protocol-report.json",
             root/"contextbench/summary.json", root/"contextbench"/(EPISODE+".json"), requests_path]
    bundle["source_files"] = {**source["source_files"], **bundle["source_files"],
                              **{str(p.resolve()): sha(p) for p in paths}}
    return source, bundle


def response_view(response):
    choice = response["choices"][0]
    message = copy.deepcopy(choice["message"])
    for call in message.get("tool_calls") or []:
        call.pop("id", None)  # Random response IDs are not semantic differences.
    return {"message": message, "finish_reason": choice["finish_reason"], "usage": response["usage"]}


def valid_bash(response):
    calls = response["message"].get("tool_calls") or []
    try:
        return len(calls) == 1 and bool(call_key(calls[0]["function"]))
    except (ValueError, TypeError, KeyError):
        return False


def replay(engine, bundle, record):
    observations = []
    for variant in ("original", "native_history", "native_history", "original"):
        request = {**copy.deepcopy(bundle["request"]), "messages": copy.deepcopy(bundle[variant])}
        start = time.perf_counter()
        response = response_view(engine.complete(request))
        observations.append({"variant": variant, "response": response, "valid_bash_call": valid_bash(response),
                             "elapsed_seconds": time.perf_counter()-start, "request_sha256": digest(request)})
        record(observations)
    grouped = {name: [o for o in observations if o["variant"] == name] for name in ("original", "native_history")}
    repeat_equal = {name: items[0]["response"] == items[1]["response"] for name, items in grouped.items()}
    same_length = all(o["response"]["usage"]["prompt_tokens"] == bundle["historical_request"]["prompt_tokens"]
                      for o in grouped["original"])
    reproduced = same_length and all(not o["response"]["message"].get("tool_calls") and
                                    o["response"]["finish_reason"] == "stop" for o in grouped["original"])
    return {"observations": observations, "repeat_equal": repeat_equal, "original_prompt_length_matches": same_length,
            "historical_non_call_failure_reproduced": reproduced,
            "native_history_valid_calls": sum(o["valid_bash_call"] for o in grouped["native_history"]),
            "format_hypothesis_supported_on_this_prompt": bool(reproduced and all(repeat_equal.values()) and
                all(o["valid_bash_call"] for o in grouped["native_history"])),
            "scope": "one saved GET input, four greedy completions, no command execution or task scoring; "
                     "reconstructed inference request, not assumed identical to the historical SDK wire request"}


def main():
    import torch
    from .io import write_json, metadata, memory, exclusive
    from .serve import Engine
    manifest = Path(sys.argv[1])
    spec = read(manifest)
    campaign = Path(spec["campaign"])
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("history audit requires one allocated BF16 GPU")
    verify_snapshot(spec["snapshot"], spec["snapshot_hash"])
    if spec["stage"] != "history_audit" or read(campaign/"config.json") != spec["config"]:
        raise ValueError("history-audit manifest differs")
    start = time.perf_counter()
    path = campaign/"history-audit.json"
    with exclusive(campaign/"audit.lock"):
        if path.exists():
            raise FileExistsError("preserve the prior diagnostic; use a new campaign")
        result = {"diagnostic_complete": False, "can_advance_to_compression": False, "contract": CONTRACT,
                  "manifest_sha256": sha(manifest), "inference_runtime": INFERENCE_RUNTIME, "error": None}
        try:
            source, bundle = prepare_source(spec["source_manifest"], verify_weights=True)
            if (bundle != read(campaign/"replay-inputs.json") or digest(bundle) != spec["inputs_sha256"] or
                    source["config"] != spec["config"]):
                raise ValueError("frozen replay inputs or source evidence changed")
            engine = Engine(bundle["artifact"], device="cuda", log=campaign/"requests.jsonl", seed=bundle["seed"])
            result.update(model=spec["config"]["model"], source_snapshot=bundle["snapshot"],
                query=bundle["query"], historical_weight_sha256=bundle["weight_sha256"],
                changed_historical_calls=len(bundle["changes"]), source_inputs_sha256=spec["inputs_sha256"])
            result.update(replay(engine, bundle, lambda rows: write_json(campaign/"partial-responses.json", rows)))
            result["diagnostic_complete"] = True
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            result.update(elapsed_seconds=time.perf_counter()-start, memory=memory(), metadata=metadata())
            write_json(path, result)
        print("HISTORY_AUDIT_COMPLETE: diagnostic only; no command execution or gate change", flush=True)


if __name__ == "__main__":
    main()
