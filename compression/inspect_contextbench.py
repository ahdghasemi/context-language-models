#!/usr/bin/env python3
"""Read saved ContextBench replies without loading a model or changing a campaign."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def messages(path):
    value = read(path)
    result = value if isinstance(value, list) else value["messages"]
    if not isinstance(result, list):
        raise ValueError(f"Expected message list in {path}")
    return result


def sources(agent, count):
    """Prefer early snapshots: a CLM final transcript may have discarded turns."""
    snapshots = sorted((agent/"context_snapshots").glob("turn-*.json"))
    trajectory = agent/"trajectory.json"
    if not snapshots and not trajectory.is_file():
        raise FileNotFoundError(f"No trajectory or context snapshots in {agent}")
    final = trajectory if trajectory.is_file() else snapshots[-1]
    early = final
    for path in snapshots:
        if sum(m.get("role") == "assistant" for m in messages(path)) >= count:
            early = path
            break
    return [("early retained context", early, slice(None, count)),
            ("final retained context", final, slice(-count, None))]


def excerpt(value, cap):
    text = json.dumps(value, ensure_ascii=True)
    if len(text) > cap:
        half = cap//2
        return text[:half]+f" ... [{len(text)-2*half} characters omitted] ... "+text[-half:]
    return text


def request_diagnostic(path, record, max_tokens=None):
    """Read usage only. Do not infer token/message alignment after context edits."""
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    totals = {"model_calls": len(rows)}
    for key in ("prompt_tokens", "completion_tokens", "tool_calls"):
        values = [r[key] for r in rows]
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"invalid {key} in {path}")
        totals[key] = sum(values)
    for key, value in totals.items():
        if key in record and record[key] != value:
            raise ValueError(f"request log {key} differs from episode record: {path}")
    if "weight_sha256" in record and any(r.get("weight_sha256") != record["weight_sha256"] for r in rows):
        raise ValueError(f"request log weights differ from episode record: {path}")
    lengths = [i for i, r in enumerate(rows) if r.get("finish_reason") == "length"]
    tools = [i for i, r in enumerate(rows) if r["tool_calls"] > 0]
    # Include transitions even when they fall outside the first/last samples.
    indices = sorted(set(range(min(3, len(rows)))) | set(range(max(0, len(rows)-2), len(rows)))
                     | set(lengths[:1]) | set(tools[:1]))
    fields = ("finish_reason", "prompt_tokens", "completion_tokens", "tool_calls", "elapsed_seconds")
    return {"request_file_sha256": hashlib.sha256(raw).hexdigest(), "totals": totals,
        "finish_reasons": dict(Counter(r.get("finish_reason", "missing") for r in rows)),
        "configured_max_tokens": max_tokens,
        "requests_at_output_limit": sum(r["completion_tokens"] == max_tokens for r in rows) if max_tokens else None,
        "first_length_request": lengths[0]+1 if lengths else None,
        "first_parsed_tool_request": tools[0]+1 if tools else None,
        "samples": [{"request_number": i+1, **{k: rows[i].get(k) for k in fields}} for i in indices],
        "scope": "server tool_calls count parsed replies, not confirmed executions; snapshot excerpts are not aligned to request numbers"}


def inspect(root, name, count=2, chars=1400, max_tokens=None):
    agent = root/"trials"/name/"agent"
    print(f"\n=== {name} ===")
    record = read(root/(name+".json"))
    print("EPISODE", json.dumps({key: record.get(key) for key in (
        "status", "rewards", "model_calls", "tool_calls", "error", "diagnostic_protocol")}))
    for filename, fields in (
        ("interactive_stream.json", ("terminated_by", "n_ops_delivered", "n_ops_total", "n_refusals", "n_explains")),
        ("usage.json", ("n_bash", "n_commands", "n_lm_calls", "n_llm_calls", "n_ctx_syncs_real", "hit_lm_call_cap", "budget_finalized", "terminated_by")),
    ):
        path = agent/filename
        if not path.is_file():
            print("MISSING", path)
            continue
        value = read(path)
        print(filename, json.dumps({key: value[key] for key in fields if key in value}))
    requests = root/(name+".requests.jsonl")
    if requests.is_file():
        diagnostic = request_diagnostic(requests, record, max_tokens)
        print("SERVER_FINISH_REASONS", json.dumps(diagnostic["finish_reasons"]))
        print("SERVER_REQUEST_DIAGNOSTIC", json.dumps(diagnostic))
    else:
        print("MISSING", requests)
    for label, path, selected in sources(agent, count):
        context = messages(path)
        indices = [i for i, m in enumerate(context) if m.get("role") == "assistant"][selected]
        print(label, "SOURCE", path)
        if not indices:
            print("No assistant messages in this saved context")
        for index in indices:
            print("ASSISTANT", excerpt(context[index], chars))
            # A reminder or tool result helps distinguish text from executed actions.
            if index+1 < len(context) and context[index+1].get("role") in {"tool", "user"}:
                print("FOLLOWING", excerpt(context[index+1], chars))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--episode", action="append", required=True,
                        help="Exact mode--task name; repeat for additional episodes")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--chars", type=int, default=1400)
    args = parser.parse_args()
    if args.samples < 1 or args.chars < 100:
        parser.error("samples must be positive and chars must be >=100")
    spec = read(args.campaign/"manifest.json")
    single_stages = {"backbone_screen", "protocol_screen", "protocol_validation"}
    if spec.get("stage") in single_stages:
        if args.run_id is not None:
            parser.error("this campaign has no training run ID; omit --run-id")
        locations = [(spec["stage"], args.campaign/"contextbench")]
    else:
        if "rows" not in spec:
            parser.error("unsupported campaign stage or missing run rows")
        locations = [(r["run_id"], args.campaign/"runs"/r["run_id"]/"contextbench")
                     for r in spec["rows"] if args.run_id is None or r["run_id"] == args.run_id]
    if not locations:
        parser.error("run-id is not in the campaign")
    print("Read-only retained-message excerpts; not raw token IDs or a full execution history.")
    for run_id, root in locations:
        summary = read(root/"summary.json")
        if summary.get("diagnostic_protocol") != spec.get("diagnostic_protocol"):
            parser.error("campaign and summary prompt-protocol identities differ")
        allowed = {e["mode"]+"--"+e["episode"] for e in summary["episodes"]}
        if not set(args.episode) <= allowed:
            parser.error(f"Unknown episode; available: {sorted(allowed)}")
        print("RUN", run_id, "COMMIT", spec["commit"])
        print("PROMPT_PROTOCOL", json.dumps(spec.get("diagnostic_protocol")))
        cap = summary.get("contextbench", {}).get("max_tokens")
        for name in args.episode:
            # Refuse a different per-episode identity rather than label it with
            # the campaign's prompt. No requirement for a passing quality gate.
            record = read(root/(name+".json"))
            if record.get("diagnostic_protocol") != summary.get("diagnostic_protocol"):
                parser.error("episode and summary prompt-protocol identities differ")
            if record not in summary["episodes"]:
                parser.error("episode record differs from summary")
            inspect(root, name, args.samples, args.chars, max_tokens=cap)


if __name__ == "__main__":
    main()
