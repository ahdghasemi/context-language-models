"""Post-hoc evidence must not turn input text, duplicates or failed commands into success."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("kv_audit", Path(__file__).resolve().parents[1]/"audit_kv_responses.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def get(qid, key):
    return {"role": "user", "content": f"=== get {qid} (0 op(s) remaining after this) ===\nGET {key}\n(Report the value.)"}


def call(cid):
    return {"role": "assistant", "tool_calls": [{"id": cid, "function": {
        "name": "bash", "arguments": json.dumps({"command": "saved command; never executed"})}}]}


def result(cid, value, code=0):
    return {"role": "tool", "tool_call_id": cid, "content": f"{value}\nREADY_FOR_NEXT_OP\n\n(exit_code={code})\n[context: 500 tokens]"}


QUERIES = {0: {"qid": 0, "key": "K00001", "value": "alpha beta #12345678"},
           1: {"qid": 1, "key": "K00002", "value": "different value #98765432"}}


def test_counts_raw_retrieval_separately_from_answer_and_ignores_repeated_frames(tmp_path):
    q = QUERIES[0]
    messages = [get(0, q["key"]), call("raw"), result("raw", q["value"])]
    save(tmp_path/"context_snapshots/turn-0001.json", {"messages": messages})
    save(tmp_path/"trajectory.json", messages)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    r = audit.collect(tmp_path, QUERIES, {0, 1})
    row = r["queries"][0]
    assert row["observed_calls"] == row["successful_calls"] == 1
    assert row["exact_value_line_in_successful_tool_result"]
    assert not row["correct_answer_block_in_successful_tool_result"]
    assert r["queries"][1]["observed_calls"] == 0
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("case", ["input", "set_output", "wrong_query", "failed", "unknown_exit", "unpaired", "unstructured"])
def test_never_counts_unassociated_or_unsuccessful_text_as_retrieval(tmp_path, case):
    q = QUERIES[0]
    if case == "input":
        messages = [get(0, q["key"]), {"role": "user", "content": q["value"]}]
    elif case == "set_output":
        messages = [call("set"), result("set", q["value"]), get(0, q["key"])]
    elif case == "wrong_query":
        messages = [get(1, QUERIES[1]["key"]), call("wrong"), result("wrong", q["value"])]
    elif case == "failed":
        messages = [get(0, q["key"]), call("failed"), result("failed", q["value"], 1)]
    elif case == "unknown_exit":
        messages = [get(0, q["key"]), call("unknown"), {"role": "tool", "tool_call_id": "unknown", "content": q["value"]}]
    elif case == "unpaired":
        messages = [get(0, q["key"]), call("different-id"), result("other-id", q["value"])]
    else:
        messages = [get(0, q["key"]), {"role": "assistant", "content": q["value"]}]
    save(tmp_path/"trajectory.json", messages)
    r = audit.collect(tmp_path, QUERIES, {0, 1})
    assert not any(x["exact_value_line_in_successful_tool_result"] for x in r["queries"])


def test_answer_blocks_and_json_tool_status_are_supported(tmp_path):
    q = QUERIES[0]
    block = f'<<<ANSWER key={q["key"]}>>>\n{q["value"]}\n<<<ANSWER END>>>'
    messages = [get(0, q["key"]), call("a"), {"role": "tool", "tool_call_id": "a",
        "content": json.dumps({"returncode": 0, "output": block})}]
    save(tmp_path/"trajectory.json", messages)
    row = audit.collect(tmp_path, QUERIES, {0})["queries"][0]
    assert row["correct_answer_block_in_successful_tool_result"]
    assert row["exact_value_line_in_successful_tool_result"]


def test_conflicting_context_copies_are_excluded_and_reported(tmp_path):
    q = QUERIES[0]
    messages = [get(0, q["key"]), call("a"), result("a", q["value"])]
    save(tmp_path/"context_snapshots/turn-0001.json", {"messages": messages})
    messages[-1] = result("a", "edited output")
    save(tmp_path/"trajectory.json", messages)
    r = audit.collect(tmp_path, QUERIES, {0})
    assert r["conflicting_call_ids"] == ["a"]
    assert r["queries"][0]["observed_calls"] == 0


@pytest.mark.parametrize("delivered,key", [(set(), "K00001"), ({0}, "K00002")])
def test_saved_query_must_match_verified_delivery(tmp_path, delivered, key):
    save(tmp_path/"trajectory.json", [get(0, key)])
    with pytest.raises(ValueError, match="verified delivered query"):
        audit.collect(tmp_path, QUERIES, delivered)


def fixture_campaign(tmp_path):
    task = tmp_path/"task/kv-store-fixture"
    q = QUERIES[0]
    save(task/"tests/gold.json", {"queries": [q], "meta": {"task_family": "kv_store"}})
    ops = [{"index": 0, "header": "get 0", "body": f'GET {q["key"]}\n(Report the value.)'}]
    stream_file = task/"environment/stream/chunks.jsonl"
    stream_file.parent.mkdir(parents=True)
    stream_file.write_text(json.dumps(ops[0])+"\n")
    lock_path = tmp_path/"task/lock.json"
    task_hash = audit.tree_hash(task)
    save(lock_path, {"revision": audit.REVISION, "tasks": [{"path": str(task), "sha256": task_hash}]})
    campaign = tmp_path/"campaign"
    spec = {"stage": "backbone_screen", "commit": "test-revision", "contextbench_lock": str(lock_path),
        "contextbench_lock_sha256": audit.sha(lock_path), "config": {"screen": {"task_names": [task.name]},
        "contextbench": {"modes": ["clm"]}}}
    save(campaign/"manifest.json", spec)
    save(campaign/"screen-report.json", {"complete": True, "passed": False, "manifest_sha256": audit.sha(campaign/"manifest.json")})
    root = campaign/"contextbench"
    record = {"episode": task.name, "mode": "clm", "task_sha256": task_hash,
        "status": "complete", "request_hash": audit.digest(spec["config"]["contextbench"]),
        "rewards": {"reward": 0, "n_answered": 0, "grader_valid": 1}}
    save(root/"summary.json", {"all_complete": True, "episodes": [record]})
    name = "clm--"+task.name
    save(root/(name+".json"), record)
    (root/(name+".requests.jsonl")).write_text('{"finish_reason": "tool_calls"}\n')
    agent = root/"trials"/name/"agent"
    stream_sha = audit.hashlib.sha256(json.dumps(ops[0], sort_keys=True).encode()).hexdigest()
    save(agent/"interactive_stream.json", {"stream": {"sha256": stream_sha}, "n_ops_total": 1,
        "n_ops_delivered": 1, "log": [{"kind": "op", "index": 0}]})
    save(agent/"trajectory.json", [get(0, q["key"]), call("raw"), result("raw", q["value"])])
    return campaign, task, agent


def test_end_to_end_diagnostic_does_not_upgrade_reward_or_modify_inputs(tmp_path):
    campaign, _, _ = fixture_campaign(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    r = audit.audit(campaign)
    assert r["official_scores_unchanged"] and not r["can_advance_to_compression"]
    episode = r["episodes"][0]
    assert episode["exact_value_line_queries"] == 1
    assert episode["correct_tool_answer_block_queries"] == episode["official_rewards"]["reward"] == 0
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("corruption", ["task", "lock", "stream", "manifest", "incomplete", "coverage", "request"])
def test_refuses_changed_inputs_and_incomplete_run(tmp_path, corruption):
    campaign, task, agent = fixture_campaign(tmp_path)
    if corruption == "task":
        save(task/"tests/gold.json", {"queries": []})
    elif corruption == "lock":
        save(tmp_path/"task/lock.json", {})
    elif corruption == "stream":
        path = agent/"interactive_stream.json"
        value = audit.read(path)
        value["stream"]["sha256"] = "different"
        save(path, value)
    elif corruption == "manifest":
        path = campaign/"manifest.json"
        value = audit.read(path)
        value["commit"] = "different"
        save(path, value)
    elif corruption in {"coverage", "request"}:
        path = campaign/"contextbench/summary.json"
        value = audit.read(path)
        if corruption == "coverage":
            value["episodes"] = []
        else:
            value["episodes"][0]["request_hash"] = "different"
        save(path, value)
    else:
        path = campaign/"screen-report.json"
        value = audit.read(path)
        value["complete"] = False
        save(path, value)
    with pytest.raises(ValueError):
        audit.audit(campaign)
