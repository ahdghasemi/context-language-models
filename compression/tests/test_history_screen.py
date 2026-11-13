"""History intervention preserves commands, delegates execution, and cannot open a compression gate."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tiny_clm.history_audit import prepare_source, replay, digest, CONTRACT, native_history, call_key
from tiny_clm.history_format import rewrite, HistoryFormatMixin, NativeHistoryMixin, CONTROL, NATIVE, identity, audit_log
from tiny_clm.history_screen import validate_replay, verify_inputs, paired_report, ARMS, CONTRACT_SCREEN
from tiny_clm.protocol_screen import read, sha, verify_inputs as verify_tool_source
from tiny_clm.provenance import INFERENCE_RUNTIME, require_gate, verify_snapshot
from test_history_audit import fixture_source, fixture_messages, FakeEngine
from test_format_screen import save
from test_screen import git_repo

ROOT = Path(__file__).resolve().parents[2]


def replay_fixture(tmp_path):
    source = fixture_source(tmp_path)
    spec, bundle = prepare_source(source)
    root = tmp_path/"replay"
    manifest = root/"manifest.json"
    save(manifest, {"stage": "history_audit", "contract": CONTRACT, "campaign": str(root),
        "config": spec["config"], "source_manifest": str(source), "inputs_sha256": digest(bundle), "commit": "fixture"})
    save(root/"config.json", spec["config"])
    save(root/"replay-inputs.json", bundle)
    report = {**replay(FakeEngine(), bundle, lambda rows: None), "contract": CONTRACT,
        "diagnostic_complete": True, "error": None, "can_advance_to_compression": False,
        "manifest_sha256": sha(manifest), "source_inputs_sha256": digest(bundle),
        "historical_weight_sha256": bundle["weight_sha256"], "metadata": {"commit": "fixture"},
        "inference_runtime": INFERENCE_RUNTIME}
    save(root/"history-audit.json", report)
    return manifest


def paired_fixture(tmp_path):
    manifest = replay_fixture(tmp_path)
    source, bundle, files = validate_replay(manifest)
    original, lock, audit, _ = verify_tool_source(source)
    baseline = read(Path(source["campaign"])/"contextbench/summary.json")
    summaries = {}
    for name, identifier in ARMS:
        summary = copy.deepcopy(baseline)
        summary.update(screen_modes=["clm"], history_protocol=identity(identifier))
        summary["episodes"] = [e for e in summary["episodes"] if e["mode"] == "clm"]
        for ep in summary["episodes"]:
            ep.update(history_protocol=identity(identifier), history_application={"history_protocol": identity(identifier),
                "requests": 3, "changed_historical_calls": int(identifier == NATIVE), "unverified_plain_calls": 0, "sha256": "a"*64})
            ep["rewards"]["reward"] = 1.0 if identifier == NATIVE else 0.0
        summaries[name] = summary
    return source["config"], lock, audit, summaries


def test_encoder_equals_replay_and_is_idempotent_and_read_only():
    fn, original, messages = fixture_messages()
    known = {call_key(fn)}
    before = copy.deepcopy(messages)
    expected, _ = native_history(messages, {call_key(fn): "prior"})
    rewritten, changes, unknown = rewrite(messages, known, enabled=True)
    assert rewritten == expected and messages == before and len(changes) == 1 and unknown == 0
    assert rewrite(rewritten, known, enabled=True) == (rewritten, [], 0)
    assert rewrite(messages, known, enabled=False) == (messages, [], 0)
    assert rewrite(original, known, enabled=True) == (original, [], 0)


@pytest.mark.parametrize("content", ['bash {"command":"never observed"}', 'bash {not json}',
    'bash {"command":3}', 'ordinary command-like prose', 'bash {"command":"printf fixture"}\ntrailing explanation'])
def test_unverified_or_nonterminal_prose_never_becomes_a_call(content):
    fn, _, messages = fixture_messages()
    messages[2]["content"] = content
    outgoing, changes, _ = rewrite(messages, {call_key(fn)}, enabled=True)
    assert outgoing == messages and changes == []
    assert not outgoing[2].get("tool_calls")


@pytest.mark.parametrize("mixin,identifier", [(HistoryFormatMixin, CONTROL), (NativeHistoryMixin, NATIVE)])
def test_live_wrapper_observes_prior_calls_and_delegates_without_mutating_mirror(tmp_path, mixin, identifier):
    fn, original, flattened = fixture_messages()
    before = copy.deepcopy(flattened)
    received, response = [], object()
    class Parent:
        async def run(self, instruction, environment, context):
            assert (instruction, environment, context) == ("unchanged", "sandbox", "context")
            # A previous pre-call view contains a real structured history entry.
            assert await self._query_with_retry("model", original) is response
            assert await self._query_with_retry("model", flattened) is response
            # A later malformed/non-call response is kept as prose, not repaired.
            unknown = flattened+[{"role": "assistant", "content": 'bash {"command":"unknown"}'}]
            assert await self._query_with_retry("model", unknown) is response
        async def _query_with_retry(self, model, messages):
            assert model == "model"
            received.append(copy.deepcopy(messages))
            return response
    class Agent(mixin, Parent):
        logs_dir = tmp_path
    asyncio.run(Agent().run("unchanged", "sandbox", "context"))
    assert original == received[0] and flattened == before
    assert received[2][-1] == {"role": "assistant", "content": 'bash {"command":"unknown"}'}
    expected, _, _ = rewrite(flattened, {call_key(fn)}, enabled=identifier == NATIVE)
    assert received[1] == expected
    audit = audit_log(tmp_path/"history_format.jsonl", identifier)
    assert audit["requests"] == 3 and audit["changed_historical_calls"] == (2 if identifier == NATIVE else 0)
    assert audit["unverified_plain_calls"] == 1
    with pytest.raises(FileExistsError):
        asyncio.run(Agent().run("unchanged", "sandbox", "context"))


@pytest.mark.parametrize("fault", ["missing_hash", "protected", "control_changed", "order"])
def test_history_application_audit_rejects_corruption(tmp_path, fault):
    row = {"history_protocol": identity(CONTROL), "request": 1, "before_sha256": "a"*64,
        "after_sha256": "a"*64, "protected_before_sha256": "b"*64, "protected_after_sha256": "b"*64,
        "noncontent_fields_equal": True, "changed": [], "unverified_plain_calls": 0}
    path = tmp_path/"history.jsonl"
    path.write_text(json.dumps(row)+"\n")
    assert audit_log(path, CONTROL)["requests"] == 1
    if fault == "missing_hash":
        row.pop("protected_before_sha256")
    elif fault == "protected":
        row["protected_after_sha256"] = "c"*64
    elif fault == "control_changed":
        row["after_sha256"] = "c"*64
    else:
        row["request"] = 3
    path.write_text(json.dumps(row)+"\n")
    with pytest.raises(ValueError):
        audit_log(path, CONTROL)


@pytest.mark.parametrize("field,value", [("diagnostic_complete", False), ("format_hypothesis_supported_on_this_prompt", False),
    ("historical_weight_sha256", "changed"), ("manifest_sha256", "changed"), ("source_inputs_sha256", "changed")])
def test_invalid_replay_cannot_launch_full_episodes(tmp_path, field, value):
    manifest = replay_fixture(tmp_path)
    validate_replay(manifest)
    path = manifest.parent/"history-audit.json"
    report = read(path)
    report[field] = value
    save(path, report)
    with pytest.raises(ValueError):
        validate_replay(manifest)


def test_replay_and_source_are_frozen_without_writes(tmp_path):
    manifest = replay_fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    source, _, files = validate_replay(manifest, verify_weights=True)
    spec = {"stage": "history_screen", "contract": CONTRACT_SCREEN, "config": source["config"],
        "replay_manifest": str(manifest), "source_files": files}
    verify_inputs(spec)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    path = manifest.parent/"history-audit.json"
    report = read(path)
    report["observations"][0]["request_sha256"] = "changed"
    save(path, report)
    with pytest.raises(ValueError, match="request/response"):
        verify_inputs(spec)


@pytest.mark.parametrize("fault", [None, "negative", "missing", "foreign", "not_applied", "duplicate"])
def test_paired_report_checks_coverage_and_keeps_compression_closed(tmp_path, fault):
    cfg, lock, audit, summaries = paired_fixture(tmp_path)
    ep = summaries["native_history"]["episodes"][0]
    if fault == "negative":
        ep["rewards"]["reward"] = 0.0
    elif fault == "missing":
        summaries.pop("control")
    elif fault == "foreign":
        ep["history_protocol"] = identity(CONTROL)
    elif fault == "not_applied":
        ep["history_application"]["changed_historical_calls"] = 0
    elif fault == "duplicate":
        summaries["native_history"]["episodes"].append(copy.deepcopy(ep))
    result = paired_report(cfg, lock, audit, summaries)
    assert result["complete"] is (fault in (None, "negative", "not_applied"))
    assert result["passed"] is (fault is None)
    assert result["can_advance_to_compression"] is False
    if result["complete"]:
        assert len(result["comparison"]) == 2
    for kind in ("baseline", "pilot"):
        with pytest.raises(ValueError):
            require_gate(result, cfg, kind)


def test_driver_dry_run_and_dependency_chain(tmp_path, monkeypatch, git_repo):
    monkeypatch.syspath_prepend(str(ROOT/"compression"))
    module = importlib.util.spec_from_file_location("history_screen_driver", ROOT/"compression/history_screen.py")
    driver = importlib.util.module_from_spec(module)
    module.loader.exec_module(driver)
    monkeypatch.setattr(driver, "ROOT", git_repo)
    monkeypatch.setattr(sys.modules["submit"], "ROOT", git_repo)
    replay_path = replay_fixture(tmp_path)
    for key, value in {"TINY_CLM_WORK": str(tmp_path), "TINY_CLM_SNAPSHOTS": str(tmp_path/"snapshots"),
        "TINY_CLM_PARTITION": "gpu-fixture", "TINY_CLM_CPU_PARTITION": "cpu-fixture", "TINY_CLM_GPU_CONSTRAINT": "fixture",
        "TINY_CLM_GPUS": "1", "TINY_CLM_VENV": "/fixture/env", "SLURM_JOB_ID": "interactive"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(driver, "verify_environment", lambda venv, expected: expected)
    calls, real = [], subprocess.check_output
    def submit(command, **kwargs):
        if command[0] != "sbatch":
            return real(command, **kwargs)
        assert "SLURM_JOB_ID" not in kwargs["env"]
        calls.append(command)
        return str(100+len(calls))+"\n"
    monkeypatch.setattr(subprocess, "check_output", submit)
    campaign = tmp_path/"paired"
    args = ["--replay-manifest", str(replay_path), "--campaign", str(campaign)]
    driver.main(args)
    assert not calls and not campaign.exists()
    driver.main(args+["--submit"])
    spec = read(campaign/"manifest.json")
    verify_snapshot(spec["snapshot"], spec["snapshot_hash"])
    verify_inputs(spec)
    assert len(calls) == 3
    assert "--dependency=afterok:101" in calls[1] and "--dependency=afterany:102" in calls[2]
    assert "--time=03:00:00" in calls[1] and "--gpus=1" in calls[1]
    assert all("--gpus=1" not in calls[i] for i in (0, 2))
    with pytest.raises(FileExistsError):
        driver.main(args+["--submit"])
