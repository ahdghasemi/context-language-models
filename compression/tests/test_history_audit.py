"""History ablation is inference-only, provenance-checked and isolated from gates."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from clm_harness.context_utils.context_string import render_editable, parse_back
from tiny_clm.history_audit import (native_history, call_key, first_get_input, prepare_source,
    replay, digest, EPISODE)
from tiny_clm.answer_format import TOOL_FORMAT_ID, protocol_identity, protocol_text
from tiny_clm.protocol_screen import validate_reference, evidence_files, compare_report, sha
from tiny_clm.provenance import verify_snapshot, require_gate
from test_tool_format_screen import source_validation, new_summary
from test_format_screen import save
from test_screen import git_repo

ROOT = Path(__file__).resolve().parents[2]


def fixture_messages():
    fn = {"name": "bash", "arguments": json.dumps({"command": "printf fixture"})}
    prefix = [{"role": "system", "content": "Use supplied tools."}, {"role": "user", "content": "Task."}]
    original = prefix+[{"role": "assistant", "content": "Thought: store.",
        "tool_calls": [{"id": "prior", "type": "function", "function": fn}]},
        {"role": "tool", "tool_call_id": "prior", "content": "fixture\n(exit_code=0)"}]
    flattened = parse_back(render_editable(original), prefix)
    flattened.append({"role": "user", "content": "=== get 0 (23 op(s) remaining after this) ===\nGET K00001"})
    return fn, original, flattened


def save_snapshots(agent):
    fn, original, flattened = fixture_messages()
    for step, messages in enumerate((original[:2], original, flattened)):
        save(agent/"context_snapshots"/f"turn-{step:04d}.json", {"kind": "agent", "step": step, "messages": messages})
    return fn, flattened


def fixture_source(tmp_path):
    validation, _ = source_validation(tmp_path)
    original, lock, audit, baseline, cfg = validate_reference(validation, TOOL_FORMAT_ID)
    root = tmp_path/"tool-screen"
    source = {"stage": "protocol_screen", "campaign": str(root), "config": cfg,
        "source_manifest": str(validation), "source_files": evidence_files(validation, original, TOOL_FORMAT_ID),
        "diagnostic_protocol": protocol_identity(TOOL_FORMAT_ID), "venv": original["venv"],
        "environment": original["environment"], "commit": "fixture"}
    manifest = root/"manifest.json"
    save(manifest, source)
    save(root/"config.json", cfg)
    (root/"reminder.txt").write_text(protocol_text(TOOL_FORMAT_ID))
    save(root/"checks.json", {"passed": True, "diagnostic_protocol": protocol_identity(TOOL_FORMAT_ID)})
    summary = new_summary(baseline, cfg)
    for ep in summary["episodes"]:
        ep["model_calls"] = 3
        ep["rewards"]["reward"] = 0
        save(root/"contextbench"/(ep["mode"]+"--"+ep["episode"]+".json"), ep)
    save(root/"contextbench/summary.json", summary)
    report = compare_report(cfg, lock, audit, baseline, summary, diagnostic_protocol=TOOL_FORMAT_ID)
    report["manifest_sha256"] = sha(manifest)
    assert report["complete"] and not report["passed"]
    save(root/"protocol-report.json", report)
    save_snapshots(root/"contextbench/trials"/EPISODE/"agent")
    rows = [{"prompt_tokens": 100, "tool_calls": 1 if i < 2 else 0,
        "finish_reason": "tool_calls" if i < 2 else "stop", "weight_sha256": audit["weight_sha256"]} for i in range(3)]
    (root/"contextbench"/(EPISODE+".requests.jsonl")).write_text("\n".join(map(json.dumps, rows)))
    return manifest


def test_native_history_matches_upstream_flattening_and_preserves_everything_else():
    fn, original, flattened = fixture_messages()
    assert not flattened[2].get("tool_calls") and 'bash {"command":' in flattened[2]["content"]
    before = copy.deepcopy(flattened)
    native, changes = native_history(flattened, {call_key(fn): "earlier snapshot"})
    assert flattened == before and len(changes) == 1
    assert native[:2] == flattened[:2] and native[3:] == flattened[3:]
    assert [m["role"] for m in native] == [m["role"] for m in flattened]
    assert not native[2].get("tool_calls")  # Historical text never becomes an executable action.
    body = native[2]["content"].split("<tool_call>\n")[1].split("\n</tool_call>")[0]
    assert json.loads(body) == {"name": "bash", "arguments": json.loads(fn["arguments"])}
    with pytest.raises(ValueError, match="no matching"):
        native_history(flattened, {})


def test_pre_call_selection_uses_only_earlier_evidence_and_is_read_only(tmp_path):
    _, flattened = save_snapshots(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    bundle = first_get_input(tmp_path)
    assert bundle["original"] == flattened and bundle["snapshot_step"] == 2
    assert bundle["query"] == {"qid": 0, "key": "K00001"}
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    # A later structured copy must not retroactively certify a prior plain command.
    path = tmp_path/"context_snapshots/turn-0001.json"
    value = json.loads(path.read_text())
    value["kind"] = "final"
    save(path, value)
    with pytest.raises(ValueError, match="no matching"):
        first_get_input(tmp_path)


@pytest.mark.parametrize("fault", ["already_answered", "step", "no_calls"])
def test_invalid_saved_prompt_fails(tmp_path, fault):
    save_snapshots(tmp_path)
    path = tmp_path/"context_snapshots/turn-0002.json"
    value = json.loads(path.read_text())
    if fault == "already_answered":
        value["messages"].append({"role": "assistant", "content": "answer"})
    elif fault == "step":
        value["step"] = 99
    else:
        value["messages"][2]["content"] = "No prior command"
    save(path, value)
    with pytest.raises(ValueError):
        first_get_input(tmp_path)


def test_source_chain_and_input_mutations(tmp_path):
    manifest = fixture_source(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    source, bundle = prepare_source(manifest, verify_weights=True)
    assert len(bundle["changes"]) == 1 and bundle["request"]["max_tokens"] == 1024
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    path = Path(bundle["snapshot"])
    value = json.loads(path.read_text())
    value["messages"][0]["content"] += "\nextra retained text"
    save(path, value)
    _, changed = prepare_source(manifest)
    assert digest(changed) != digest(bundle)
    record = manifest.parent/"contextbench"/(EPISODE+".json")
    value = json.loads(record.read_text())
    value["weight_sha256"] = "changed"
    save(record, value)
    with pytest.raises(ValueError, match="source episode"):
        prepare_source(manifest)


class FakeEngine:
    def __init__(self, *, native=True, prompt_tokens=100):
        self.requests = []
        self.native, self.prompt_tokens = native, prompt_tokens
    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        native = "<tool_call>" in json.dumps(request["messages"]) and self.native
        message = {"role": "assistant", "content": "text"}
        if native:
            message["tool_calls"] = [{"id": str(len(self.requests)), "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"not executed"}'}}]
        return {"choices": [{"message": message, "finish_reason": "tool_calls" if native else "stop"}],
                "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 4}}


@pytest.mark.parametrize("native,prompt_tokens,supported", [(True, 100, True), (False, 100, False), (True, 101, False)])
def test_replay_four_completions_without_scoring_or_executing(tmp_path, native, prompt_tokens, supported):
    manifest = fixture_source(tmp_path)
    source, bundle = prepare_source(manifest)
    engine, checkpoints = FakeEngine(native=native, prompt_tokens=prompt_tokens), []
    result = replay(engine, bundle, lambda rows: checkpoints.append(copy.deepcopy(rows)))
    assert len(engine.requests) == 4 and [len(c) for c in checkpoints] == [1, 2, 3, 4]
    assert result["repeat_equal"] == {"original": True, "native_history": True}
    assert result["format_hypothesis_supported_on_this_prompt"] is supported
    assert all({k: v for k, v in req.items() if k != "messages"} == bundle["request"] for req in engine.requests)
    for gate in ("baseline", "pilot"):
        with pytest.raises(ValueError):
            require_gate(result, source["config"], gate)


def test_single_job_submission_snapshot_and_dry_run(tmp_path, monkeypatch, git_repo):
    monkeypatch.syspath_prepend(str(ROOT/"compression"))
    spec = importlib.util.spec_from_file_location("history_audit_driver", ROOT/"compression/history_audit.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, "ROOT", git_repo)
    monkeypatch.setattr(sys.modules["submit"], "ROOT", git_repo)
    source = fixture_source(tmp_path)
    for key, value in {"TINY_CLM_WORK": str(tmp_path), "TINY_CLM_SNAPSHOTS": str(tmp_path/"snapshots"),
        "TINY_CLM_PARTITION": "gpu-fixture", "TINY_CLM_GPU_CONSTRAINT": "fixture", "TINY_CLM_GPUS": "1",
        "TINY_CLM_VENV": "/fixture/env", "SLURM_JOB_ID": "parent"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(driver, "verify_environment", lambda venv, expected: expected)
    calls, real = [], subprocess.check_output
    def submit(command, **kwargs):
        if command[0] != "sbatch":
            return real(command, **kwargs)
        assert "SLURM_JOB_ID" not in kwargs["env"]
        calls.append(command)
        return "123\n"
    monkeypatch.setattr(subprocess, "check_output", submit)
    campaign = tmp_path/"audit"
    args = ["--source-manifest", str(source), "--campaign", str(campaign)]
    driver.main(args)
    assert not calls and not campaign.exists()
    driver.main(args+["--submit"])
    record = json.loads((campaign/"manifest.json").read_text())
    verify_snapshot(record["snapshot"], record["snapshot_hash"])
    assert digest(json.loads((campaign/"replay-inputs.json").read_text())) == record["inputs_sha256"]
    assert len(calls) == 1 and "--time=00:30:00" in calls[0] and "--gpus=1" in calls[0]
    assert not any("--array=" in a for a in calls[0])
    with pytest.raises(FileExistsError):
        driver.main(args+["--submit"])
