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
    monkeypatch.setattr(sys, "argv", ["inspect_contextbench.py", "--campaign", str(tmp_path), "--episode", "clm--fixture"])
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    inspector.main()
    assert "OBSERVED_REPLY" in capsys.readouterr().out
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
    assert [r["request_number"] for r in result["samples"]] == [1, 2, 3, 6, 8, 9]
    assert path.read_bytes() == before
    for key, changed in (("model_calls", 8), ("tool_calls", 0), ("weight_sha256", "wrong")):
        with pytest.raises(ValueError, match="differs|differ"):
            inspector.request_diagnostic(path, {**record, key: changed}, 1024)


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
