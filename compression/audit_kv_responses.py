#!/usr/bin/env python3
"""Read-only post-hoc KV response audit; never replaces an official reward.

Uses only the standard library, saved messages, and verified task inputs. It
does not execute saved commands, load a model, or write into a campaign.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


REVISION = "fb533c18c78a82d3f1b2047a9c1700735cf829ae"
# Match delivered operation headers, not mentions of GET in instructions/code.
GET = re.compile(r"^=== get (\d+) \(\d+ op\(s\) remaining after this\) ===\s*\nGET (K\d+)\s*$", re.MULTILINE)
# Same ANSWER syntax as the pinned ContextBench grader, not a relaxed grader.
# See third_party/NOTICE.md for attribution and docs/KV_RESPONSE_AUDIT.md.
ANSWER = re.compile(r"<<<\s*ANSWER\s+key\s*=\s*([^\s>]+)\s*>>>(.*?)<<<\s*ANSWER\s+END\s*>>>", re.DOTALL | re.IGNORECASE)
EXIT = re.compile(r"^(?:\(exit_code=(-?\d+)\)|exit_code:\s*(-?\d+))\s*$", re.MULTILINE)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tree_hash(path):
    return digest({str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*")) if p.is_file()})


def normalize(text):
    return " ".join(text.split())


def tool_text(message):
    text = message.get("content") or ""
    if not isinstance(text, str):
        raise ValueError("tool content is not text")
    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        text = "\n".join(v for v in obj.values() if isinstance(v, str))
        codes = [obj[k] for k in ("returncode", "return_code", "exit_code") if k in obj]
    else:
        codes = [int(a or b) for a, b in EXIT.findall(text)]
    # Missing/contradictory status is unknown; never assume execution success.
    code = codes[0] if codes and all(type(c) is int and c == codes[0] for c in codes) else None
    return text, code


def collect(agent, queries, delivered):
    paths = sorted((agent/"context_snapshots").glob("turn-*.json"))
    trajectory = agent/"trajectory.json"
    if trajectory.is_file():
        paths.append(trajectory)
    if not paths:
        raise FileNotFoundError("no saved context snapshots or trajectory")
    calls, unpaired = {}, set()
    inventory = {}
    for path in paths:
        inventory[str(path.relative_to(agent))] = sha(path)
        value = read(path)
        messages = value if isinstance(value, list) else value["messages"]
        active, pending = None, {}
        for message in messages:
            role, content = message.get("role"), message.get("content") or ""
            if role == "user" and isinstance(content, str):
                matches = GET.findall(content)
                if matches:
                    qid, key = matches[-1]
                    active = int(qid)
                    if active not in queries or queries[active]["key"] != key or active not in delivered:
                        raise ValueError("saved GET does not match the verified delivered query")
            elif role == "assistant":
                for call in message.get("tool_calls") or []:
                    fn = call.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            continue
                    if (fn.get("name") == "bash" and isinstance(args, dict)
                            and isinstance(args.get("command"), str) and call.get("id")):
                        pending[call["id"]] = (active, args["command"])
            elif role == "tool":
                cid = message.get("tool_call_id")
                if cid not in pending:
                    unpaired.add(cid or digest(message))
                    continue
                qid, command = pending.pop(cid)
                if qid is None:
                    continue  # SET ingestion is not evidence of a GET response.
                text, code = tool_text(message)
                observation = {"qid": qid, "command_sha256": digest(command),
                    "result_sha256": digest(message), "exit_code": code,
                    "exact_value_line": normalize(queries[qid]["value"]) in {normalize(x) for x in text.splitlines()},
                    "correct_answer_block": any(key == queries[qid]["key"] and
                        normalize(queries[qid]["value"]) in normalize(val) for key, val in ANSWER.findall(text))}
                item = calls.setdefault(cid, {"first_source": str(path.relative_to(agent)), "observations": {}})
                item["observations"][digest(observation)] = observation
    rows = []
    for qid, query in sorted(queries.items()):
        # Context edits can change retained messages. Conflicting observations
        # are excluded, rather than choosing whichever copy looks successful.
        selected = [next(iter(c["observations"].values())) for c in calls.values()
                    if len(c["observations"]) == 1 and next(iter(c["observations"].values()))["qid"] == qid]
        good = [o for o in selected if o["exit_code"] == 0]
        rows.append({"qid": qid, "key": query["key"], "delivered": qid in delivered,
            "observed_calls": len(selected), "successful_calls": len(good),
            "failed_calls": sum(o["exit_code"] not in (0, None) for o in selected),
            "unknown_exit_calls": sum(o["exit_code"] is None for o in selected),
            "exact_value_line_in_successful_tool_result": any(o["exact_value_line"] for o in good),
            "correct_answer_block_in_successful_tool_result": any(o["correct_answer_block"] for o in good)})
    # Sources and hashes make the diagnostic reproducible without printing gold
    # values or model-generated shell commands into the report.
    return {"queries": rows, "source_files": inventory,
        "conflicting_call_ids": sorted(cid for cid, c in calls.items() if len(c["observations"]) > 1),
        "unpaired_tool_result_ids": sorted(unpaired),
        "paired_calls": [{"call_id": cid, **item, "observations": list(item["observations"].values())}
                         for cid, item in sorted(calls.items())]}


def audit(campaign):
    manifest_path = campaign/"manifest.json"
    spec = read(manifest_path)
    if spec.get("stage") != "backbone_screen":
        raise ValueError("this diagnostic requires a completed backbone screen")
    report = read(campaign/"screen-report.json")
    if not report.get("complete") or report.get("manifest_sha256") != sha(manifest_path):
        raise ValueError("screen is incomplete or its manifest identity differs")
    lock_path = Path(spec["contextbench_lock"])
    if sha(lock_path) != spec["contextbench_lock_sha256"]:
        raise ValueError("task lock changed after submission")
    lock = read(lock_path)
    if lock["revision"] != REVISION:
        raise ValueError("unsupported ContextBench revision")
    root = campaign/"contextbench"
    summary = read(root/"summary.json")
    if not summary.get("all_complete"):
        raise ValueError("ContextBench episodes are incomplete")
    expected = {(mode, name) for mode in spec["config"]["contextbench"]["modes"]
                for name in spec["config"]["screen"]["task_names"]}
    actual = [(r["mode"], r["episode"]) for r in summary["episodes"]]
    if not expected or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("episode coverage differs from the declared screen")
    output = []
    for record in summary["episodes"]:
        if record.get("status") != "complete" or record.get("rewards", {}).get("grader_valid") != 1:
            raise ValueError("episode is not complete with a valid official grade")
        if record.get("request_hash") != digest(spec["config"]["contextbench"]):
            raise ValueError("episode configuration differs from the manifest")
        name, mode = record["episode"], record["mode"]
        if name not in spec["config"]["screen"]["task_names"] or mode not in spec["config"]["contextbench"]["modes"]:
            raise ValueError("episode outside the declared screen")
        tasks = [t for t in lock["tasks"] if Path(t["path"]).name == name]
        if len(tasks) != 1:
            raise ValueError("task missing or ambiguous in lock")
        task = Path(tasks[0]["path"])
        if tree_hash(task) != tasks[0]["sha256"] or record["task_sha256"] != tasks[0]["sha256"]:
            raise ValueError("task files changed")
        episode_name = mode+"--"+name
        if read(root/(episode_name+".json")) != record:
            raise ValueError("episode record differs from summary")
        agent = root/"trials"/episode_name/"agent"
        gold = read(task/"tests/gold.json")
        queries = {int(q["qid"]): q for q in gold["queries"]}
        if not queries or len(queries) != len(gold["queries"]) or gold["meta"]["task_family"] != "kv_store":
            raise ValueError("expected nonempty KV queries with unique IDs")
        ops = [json.loads(line) for line in (task/"environment/stream/chunks.jsonl").read_text().splitlines() if line.strip()]
        stream = read(agent/"interactive_stream.json")
        stream_sha = hashlib.sha256("\n".join(json.dumps(r, sort_keys=True) for r in ops).encode()).hexdigest()
        if stream["stream"]["sha256"] != stream_sha or stream["n_ops_total"] != len(ops):
            raise ValueError("saved operation stream identity differs from task")
        indices = [entry["index"] for entry in stream["log"] if entry["kind"] == "op"]
        if indices != list(range(stream["n_ops_delivered"])) or len(indices) > len(ops):
            raise ValueError("operation delivery log is inconsistent")
        delivered = set()
        for index in indices:
            op = ops[index]
            match = re.fullmatch(r"get (\d+)", op["header"])
            if match:
                qid = int(match[1])
                if qid not in queries or op["body"].splitlines()[0] != "GET "+queries[qid]["key"]:
                    raise ValueError("gold query and operation disagree")
                delivered.add(qid)
        result = collect(agent, queries, delivered)
        rows = result["queries"]
        requests = root/(episode_name+".requests.jsonl")
        result.update(episode=episode_name, official_rewards=record["rewards"],
            total_queries=len(rows), delivered_queries=len(delivered),
            queries_with_observed_response=sum(r["observed_calls"] > 0 for r in rows),
            exact_value_line_queries=sum(r["exact_value_line_in_successful_tool_result"] for r in rows),
            correct_tool_answer_block_queries=sum(r["correct_answer_block_in_successful_tool_result"] for r in rows),
            finish_reasons=dict(Counter(json.loads(line).get("finish_reason", "missing")
                                for line in requests.read_text().splitlines() if line.strip())),
            task_sha256=tasks[0]["sha256"], gold_sha256=sha(task/"tests/gold.json"),
            stream_record_sha256=sha(agent/"interactive_stream.json"), requests_sha256=sha(requests))
        output.append(result)
    return {"scope": "post-hoc response diagnostic; not a benchmark reward or gate; no model inference",
        "official_scores_unchanged": True, "can_advance_to_compression": False,
        "campaign_commit": spec["commit"], "manifest_sha256": sha(manifest_path),
        "audit_script_sha256": sha(Path(__file__)), "episodes": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--details", action="store_true", help="include every query, source hash and paired call")
    args = parser.parse_args()
    result = audit(args.campaign)
    if not args.details:
        for episode in result["episodes"]:
            for key in ("queries", "source_files", "paired_calls"):
                episode.pop(key)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
