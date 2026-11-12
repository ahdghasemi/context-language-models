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
