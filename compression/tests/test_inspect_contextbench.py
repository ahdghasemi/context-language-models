"""Exercise both upstream log layouts and preserve the original campaign bytes."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("inspect_contextbench",
    Path(__file__).resolve().parents[1]/"inspect_contextbench.py")
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


@pytest.mark.parametrize("mode", ["clm", "append_only"])
def test_reads_early_and_final_context_without_mutation(tmp_path, capsys, mode):
    name = mode+"--fixture"
    agent = tmp_path/"trials"/name/"agent"
    snapshots = agent/"context_snapshots"
    snapshots.mkdir(parents=True)
    def save(path, value):
        path.write_text(json.dumps(value))
    save(tmp_path/(name+".json"), {"status": "complete", "rewards": {"reward": 0}})
    save(agent/"interactive_stream.json", {"terminated_by": "max_steps"})
    save(agent/"usage.json", {"n_bash": 0})
    early = [{"role": "assistant", "content": "EARLY_REPLY"},
             {"role": "user", "content": "NO_TOOL_REMINDER"}]
    final = [{"role": "assistant", "content": "LATE_REPLY"}]
    save(snapshots/"turn-0000.json", {"messages": [{"role": "user", "content": "task"}]})
    save(snapshots/"turn-0001.json", {"messages": early})
    save(snapshots/"turn-0002.json", {"messages": final})
    if mode == "clm":
        save(agent/"trajectory.json", final)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    inspector.inspect(tmp_path, name, count=1)
    output = capsys.readouterr().out
    assert "EARLY_REPLY" in output and "LATE_REPLY" in output
    assert "NO_TOOL_REMINDER" in output and "max_steps" in output
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_missing_transcripts_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trajectory or context snapshots"):
        inspector.sources(tmp_path, 2)


def test_long_excerpts_keep_head_and_tail_with_explicit_truncation():
    result = inspector.excerpt("HEAD"+"x"*200+"TAIL", 100)
    assert "HEAD" in result and "TAIL" in result and "characters omitted" in result


@pytest.mark.parametrize("stage", ["baseline", "backbone_screen", "protocol_screen", "protocol_validation"])
def test_cli_locates_screen_and_baseline_transcripts(tmp_path, monkeypatch, capsys, stage):
    import sys
    spec = {"stage": stage, "commit": "fixture", "rows": [{"run_id": "original"}]}
    protocol = {"id": "fixture-format", "suffix_sha256": "a"*64} if stage.startswith("protocol_") else None
    if protocol:
        spec["diagnostic_protocol"] = protocol
    (tmp_path/"manifest.json").write_text(json.dumps(spec))
    root = tmp_path/"contextbench" if stage != "baseline" else tmp_path/"runs/original/contextbench"
    agent = root/"trials/clm--fixture/agent"
    agent.mkdir(parents=True)
    record = {"mode": "clm", "episode": "fixture", "status": "complete", "rewards": {"reward": 0},
              "diagnostic_protocol": protocol}
    (root/"summary.json").write_text(json.dumps({"episodes": [record], "diagnostic_protocol": protocol}))
    (root/"clm--fixture.json").write_text(json.dumps(record))
    (agent/"trajectory.json").write_text(json.dumps([{"role": "assistant", "content": "OBSERVED_REPLY"}]))
    monkeypatch.setattr(sys, "argv", ["inspect_contextbench.py", "--campaign", str(tmp_path),
        "--episode", "clm--fixture", "--get-samples", "2"])
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    inspector.main()
    output = capsys.readouterr().out
    assert "OBSERVED_REPLY" in output
    assert "No delivered GET header found" in output
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_request_diagnostic_exposes_length_and_tool_transitions_without_executing_content(tmp_path):
    path = tmp_path/"requests.jsonl"
    rows = [{"finish_reason": "stop", "prompt_tokens": 100+i, "completion_tokens": 6,
             "tool_calls": 0, "elapsed_seconds": 0.1, "weight_sha256": "same"} for i in range(9)]
    rows[0].update(finish_reason="length", completion_tokens=1024)
    rows[5].update(finish_reason="tool_calls", tool_calls=1)
    path.write_text("\n".join(json.dumps(r) for r in rows))
    record = {"model_calls": 9, "tool_calls": 1, "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
              "completion_tokens": sum(r["completion_tokens"] for r in rows), "weight_sha256": "same"}
    before = path.read_bytes()
    result = inspector.request_diagnostic(path, record, 1024)
    assert result["first_length_request"] == 1 and result["first_parsed_tool_request"] == 6
    assert result["requests_at_output_limit"] == 1
    assert result["finish_reasons"] == {"length": 1, "stop": 7, "tool_calls": 1}
    assert result["last_parsed_tool_request"] == 6
    assert result["first_no_tool_after_tool_request"] == 7
    assert [r["request_number"] for r in result["samples"]] == [1, 2, 3, 6, 7, 8, 9]
    assert path.read_bytes() == before
    for key, changed in (("model_calls", 8), ("tool_calls", 0), ("weight_sha256", "wrong")):
        with pytest.raises(ValueError, match="differs|differ"):
            inspector.request_diagnostic(path, {**record, key: changed}, 1024)


def test_first_get_retained_responses_excludes_other_queries_and_deduplicates(tmp_path):
    snapshots = tmp_path/"context_snapshots"
    snapshots.mkdir()
    header = {"role": "user", "content": "=== get 5 (23 op(s) remaining after this) ===\nGET K00035\n"}
    reply = {"role": "assistant", "content": "bash {\"command\":\"not executed\"}"}
    reminder = {"role": "user", "content": "No tool call."}
    second = {"role": "assistant", "content": "Waiting for the next operation."}
    other = {"role": "user", "content": "=== get 2 (22 op(s) remaining after this) ===\nGET K00010\n"}
    contexts = [
        [{"role": "user", "content": "The instruction mentions GET K99999."}],
        [header], [header, reply, reminder], [header, reply, reminder],
        [header, reply, reminder, second, reminder, other, {"role": "assistant", "content": "OTHER_QUERY"}],
        [{"role": "assistant", "content": "HEADER_EDITED_AWAY"}],
    ]
    for index, context in enumerate(contexts):
        (snapshots/f"turn-{index:04d}.json").write_text(json.dumps({"messages": context}))
    before = {p: p.read_bytes() for p in snapshots.iterdir()}
    target, source, replies = inspector.first_get_responses(tmp_path, 3)
    assert target == ("5", "K00035") and source.name == "turn-0001.json"
    assert [r[1] for r in replies] == [reply, second]
    assert before == {p: p.read_bytes() for p in snapshots.iterdir()}
    assert len(inspector.first_get_responses(tmp_path, 1)[2]) == 1


@pytest.mark.parametrize("header_present", [False, True])
def test_first_get_without_retained_answer_does_not_invent_a_response(tmp_path, header_present):
    context = [{"role": "user", "content": "=== get 0 (23 op(s) remaining after this) ===\nGET K00001"}] if header_present else []
    (tmp_path/"trajectory.json").write_text(json.dumps(context))
    target, source, replies = inspector.first_get_responses(tmp_path, 2)
    assert bool(target) == header_present
    assert bool(source) == header_present
    assert replies == []


@pytest.mark.parametrize("fault", ["summary_protocol", "episode_protocol", "record"])
def test_cli_rejects_mixed_protocols_and_changed_episode_records(tmp_path, monkeypatch, fault):
    import sys
    root = tmp_path/"contextbench"
    root.mkdir()
    protocol = {"id": "declared-format"}
    manifest = {"stage": "protocol_validation", "commit": "fixture", "diagnostic_protocol": protocol}
    record = {"mode": "clm", "episode": "fixture", "diagnostic_protocol": protocol, "status": "complete"}
    summary = {"episodes": [dict(record)], "diagnostic_protocol": protocol}
    if fault == "summary_protocol":
        summary["diagnostic_protocol"] = None
    elif fault == "episode_protocol":
        record["diagnostic_protocol"] = None
    else:
        record["status"] = "changed"
    for p, obj in [(tmp_path/"manifest.json", manifest), (root/"summary.json", summary), (root/"clm--fixture.json", record)]:
        p.write_text(json.dumps(obj))
    monkeypatch.setattr(sys, "argv", ["inspect_contextbench.py", "--campaign", str(tmp_path), "--episode", "clm--fixture"])
    with pytest.raises(SystemExit):
        inspector.main()


@pytest.fixture
def paired_campaign(tmp_path):
    import hashlib
    from tiny_clm.answer_format import TOOL_FORMAT_ID, protocol_identity
    from tiny_clm.history_format import CONTROL, NATIVE, identity
    protocol = protocol_identity(TOOL_FORMAT_ID)
    spec = {"stage": "history_screen", "contract": "paired-history-episodes-v1", "commit": "fixture",
            "config": {"screen": {"task_names": ["fixture"]}, "contextbench": {"max_tokens": 1024}}}
    (tmp_path/"manifest.json").write_text(json.dumps(spec))
    report = {"contract": spec["contract"], "commit": spec["commit"], "complete": True,
              "passed": False, "can_advance_to_compression": False,
              "manifest_sha256": hashlib.sha256((tmp_path/"manifest.json").read_bytes()).hexdigest(), "arms": {}}
    for name, identifier in (("control", CONTROL), ("native_history", NATIVE)):
        root = tmp_path/name
        agent = root/"trials/clm--fixture/agent"
        agent.mkdir(parents=True)
        log = b'{"fixture": "retained application log"}\n'
        (agent/"history_format.jsonl").write_bytes(log)
        record = {"mode": "clm", "episode": "fixture", "status": "complete", "rewards": {"reward": 0},
                  "diagnostic_protocol": protocol, "history_protocol": identity(identifier),
                  "history_application": {"history_protocol": identity(identifier), "sha256": hashlib.sha256(log).hexdigest()}}
        summary = {"episodes": [record], "diagnostic_protocol": protocol, "history_protocol": identity(identifier),
                   "screen_modes": ["clm"], "screen_task_names": ["fixture"], "contextbench": spec["config"]["contextbench"]}
        (root/"summary.json").write_text(json.dumps(summary))
        (root/"clm--fixture.json").write_text(json.dumps(record))
        report["arms"][name] = {"diagnostic_protocol": protocol,
                                "episodes": [{k: record[k] for k in ("episode", "mode", "status", "rewards")}]}
        context = [{"role": "user", "content": "=== get 0 (23 op(s) remaining after this) ===\nGET K00001\n"},
                   {"role": "assistant", "content": name+"_RETAINED_COMMAND"},
                   {"role": "tool", "content": name+"_RETAINED_RESULT"}]
        (agent/"trajectory.json").write_text(json.dumps(context))
    (tmp_path/"history-screen.json").write_text(json.dumps(report))
    return tmp_path


@pytest.mark.parametrize("arm", [None, "control", "native_history"])
def test_paired_negative_campaign_readonly_and_arm_selection(paired_campaign, monkeypatch, capsys, arm):
    import sys
    root = paired_campaign
    argv = ["inspect_contextbench.py", "--campaign", str(root), "--episode", "clm--fixture", "--get-samples", "2"]
    monkeypatch.setattr(sys, "argv", argv+(["--arm", arm] if arm else []))
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    inspector.main()
    output = capsys.readouterr().out
    for name in ("control", "native_history"):
        assert (name+"_RETAINED_RESULT" in output) == (arm is None or arm == name)
    assert "FIRST_GET_RETAINED_RESPONSES" in output and "HISTORY_PROTOCOL" in output
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["manifest", "report_commit", "report_reward", "history_protocol",
    "prompt_protocol", "scope", "duplicate_episode", "record", "application_log", "run_id"])
def test_paired_inspection_rejects_mixed_evidence(paired_campaign, monkeypatch, fault):
    import sys
    root = paired_campaign
    def alter(relative, change):
        path = root/relative
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))
    if fault == "manifest":
        alter("manifest.json", lambda x: x.update(commit="other"))
    elif fault == "report_commit":
        alter("history-screen.json", lambda x: x.update(commit="other"))
    elif fault == "report_reward":
        alter("history-screen.json", lambda x: x["arms"]["native_history"]["episodes"][0].update(rewards={"reward": 1}))
    elif fault in ("history_protocol", "prompt_protocol", "scope"):
        key = {"history_protocol": "history_protocol", "prompt_protocol": "diagnostic_protocol", "scope": "screen_modes"}[fault]
        alter("native_history/summary.json", lambda x: x.update({key: None}))
    elif fault == "duplicate_episode":
        alter("native_history/summary.json", lambda x: x["episodes"].append(x["episodes"][0]))
    elif fault == "record":
        alter("native_history/clm--fixture.json", lambda x: x.update(rewards={"reward": 1}))
    elif fault == "application_log":
        (root/"native_history/trials/clm--fixture/agent/history_format.jsonl").write_text("changed")
    argv = ["inspect_contextbench.py", "--campaign", str(root), "--arm", "native_history", "--episode", "clm--fixture"]
    monkeypatch.setattr(sys, "argv", argv+(["--run-id", "wrong"] if fault == "run_id" else []))
    with pytest.raises(SystemExit):
        inspector.main()


def test_arm_option_cannot_relabel_an_original_screen(tmp_path, monkeypatch):
    import sys
    (tmp_path/"manifest.json").write_text(json.dumps({"stage": "backbone_screen"}))
    monkeypatch.setattr(sys, "argv", ["inspect_contextbench.py", "--campaign", str(tmp_path),
        "--arm", "native_history", "--episode", "clm--fixture"])
    with pytest.raises(SystemExit):
        inspector.main()
