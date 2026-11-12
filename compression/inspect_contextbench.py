#!/usr/bin/env python3
"""Read saved ContextBench replies without loading a model or changing a campaign."""
from __future__ import annotations

import argparse
from collections import Counter
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


def inspect(root, name, count=2, chars=1400):
    agent = root/"trials"/name/"agent"
    print(f"\n=== {name} ===")
    record = read(root/(name+".json"))
    print("EPISODE", json.dumps({key: record.get(key) for key in (
        "status", "rewards", "model_calls", "tool_calls", "error")}))
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
        counts = Counter(json.loads(line).get("finish_reason", "missing")
                         for line in requests.read_text().splitlines() if line.strip())
        print("SERVER_FINISH_REASONS", json.dumps(counts))
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
    if spec.get("stage") == "backbone_screen":
        if args.run_id is not None:
            parser.error("a backbone screen has no training run ID; omit --run-id")
        locations = [("backbone_screen", args.campaign/"contextbench")]
    else:
        locations = [(r["run_id"], args.campaign/"runs"/r["run_id"]/"contextbench")
                     for r in spec["rows"] if args.run_id is None or r["run_id"] == args.run_id]
    if not locations:
        parser.error("run-id is not in the campaign")
    print("Read-only retained-message excerpts; not raw token IDs or a full execution history.")
    for run_id, root in locations:
        summary = read(root/"summary.json")
        allowed = {e["mode"]+"--"+e["episode"] for e in summary["episodes"]}
        if not set(args.episode) <= allowed:
            parser.error(f"Unknown episode; available: {sorted(allowed)}")
        print("RUN", run_id, "COMMIT", spec["commit"])
        for name in args.episode:
            inspect(root, name, args.samples, args.chars)


if __name__ == "__main__":
    main()
