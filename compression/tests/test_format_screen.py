"""One prompt-only intervention, explicit provenance, no gate or scheduler bypass."""
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tiny_clm.answer_format import (FORMAT_ID, TOOL_FORMAT_ID, FORMAT_TEXT, FormatReminderMixin, add_reminder,
    protocol_identity, agent_path, application_valid, text_sha)
from tiny_clm.protocol_screen import validate_source, evidence_files, verify_inputs, compare_report, sha
from tiny_clm.provenance import require_gate, verify_snapshot, INFERENCE_RUNTIME
from tiny_clm.screen_contract import screen_report
from test_screen import fixture, git_repo

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def application():
    return {"diagnostic_protocol": protocol_identity(FORMAT_ID),
            "original_instruction_sha256": "a"*64, "modified_instruction_sha256": "b"*64}


@pytest.mark.parametrize("mode", ["clm", "append_only"])
def test_wrapper_changes_only_instruction_and_delegates_once(tmp_path, mode):
    received = []
    class Parent:
        async def run(self, instruction, environment, context):
            received.append((instruction, environment, context))
            return context
    class Wrapped(FormatReminderMixin, Parent):
        logs_dir = tmp_path
    instruction = "# Key-Value Store - process graded\nOriginal instructions.\n"
    env, context = object(), object()
    assert asyncio.run(Wrapped().run(instruction, env, context)) is context
    assert received == [(instruction.rstrip()+"\n\n"+FORMAT_TEXT, env, context)]
    saved = json.loads((tmp_path/"answer_format_protocol.json").read_text())
    assert application_valid(saved, protocol_identity(FORMAT_ID))
    assert saved["original_instruction_sha256"] == text_sha(instruction)
    assert agent_path(mode, FORMAT_ID).startswith("tiny_clm.contextbench_protocol:")
    assert "format" not in agent_path(mode).lower()
    with pytest.raises(FileExistsError):
        asyncio.run(Wrapped().run(instruction, env, context))
    assert len(received) == 1
    for bad in ("# Different task", add_reminder(instruction)):
        with pytest.raises(ValueError):
            add_reminder(bad)
    with pytest.raises(ValueError):
        agent_path(mode, "unrecognized")


def test_shell_pipeline_can_mask_empty_lookup_without_pipefail(tmp_path):
    path = tmp_path/"batch0.txt"
    path.write_text("SET K00001 = stored value\n")
    # Fixed synthetic command; never evaluate a saved model-generated command.
    command = "grep -h 'SET K00105 =' batch0.txt | cut -d'=' -f2- && echo READY_FOR_NEXT_OP"
    env = {k: v for k, v in os.environ.items() if k not in {"BASH_ENV", "SHELLOPTS"}}
    base = ["bash", "--noprofile", "--norc"]
    result = subprocess.run(base+["-c", command], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0 and result.stdout.strip() == "READY_FOR_NEXT_OP"
    guarded = subprocess.run(base+["-o", "pipefail", "-c", command], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert guarded.returncode != 0 and not guarded.stdout


def variant_summary(summary):
    result = copy.deepcopy(summary)
    result["diagnostic_protocol"] = protocol_identity(FORMAT_ID)
    for ep in result["episodes"]:
        ep.update(diagnostic_protocol=protocol_identity(FORMAT_ID), protocol_application=application())
        ep["rewards"].update(reward=1.0, n_answered=24)
    return result


def test_variant_has_own_gate_and_cannot_be_mixed_into_original_results(monkeypatch):
    cfg, reference, lock, audit, original = fixture()
    variant = variant_summary(original)
    report = compare_report(cfg, lock, audit, original, variant)
    assert report["complete"] and report["passed"] and report["gate_type"] == "protocol_screen"
    assert not report["can_advance_to_compression"]
    assert report["comparison"][0]["original_reward"] == 0
    assert report["comparison"][0]["reminder_reward"] == 1
    assert not screen_report(cfg, lock, audit, variant)["complete"]
    assert not screen_report(cfg, lock, audit, original, diagnostic_protocol=FORMAT_ID)["complete"]
    for kind in ("baseline", "pilot"):
        with pytest.raises(ValueError):
            require_gate(report, reference["config"], kind)
    monkeypatch.syspath_prepend(str(ROOT/"compression"))
    from aggregate import validate_episodes
    with pytest.raises(ValueError, match="diagnostic prompt"):
        validate_episodes(cfg, variant)


@pytest.mark.parametrize("corruption", ["identity", "application", "missing", "cap", "weights", "grade"])
def test_variant_report_rejects_incomplete_mixed_or_changed_evidence(corruption):
    cfg, _, lock, audit, original = fixture()
    variant = variant_summary(original)
    if corruption == "identity":
        variant["episodes"][0]["diagnostic_protocol"]["suffix_sha256"] = "wrong"
    elif corruption == "application":
        variant["episodes"][0].pop("protocol_application")
    elif corruption == "missing":
        variant["episodes"].pop()
    elif corruption == "cap":
        variant["contextbench"]["max_tokens"] *= 2
    elif corruption == "weights":
        variant["episodes"][0]["weight_sha256"] = "different"
    else:
        variant["episodes"][0]["rewards"]["grader_valid"] = 0
    report = compare_report(cfg, lock, audit, original, variant)
    assert not report["complete"] and not report["passed"] and not report["comparison"]


def source_fixture(tmp_path):
    cfg, _, lock, audit, summary = fixture()
    root = tmp_path/"source"
    root.mkdir()
    lock_path = tmp_path/"contextbench-lock.json"
    save(lock_path, lock)
    weight = root/"export/model.safetensors"
    weight.parent.mkdir()
    weight.write_bytes(b"synthetic-weights")
    audit["weight_sha256"] = sha(weight)
    for ep in summary["episodes"]:
        ep["weight_sha256"] = audit["weight_sha256"]
    source = {"stage": "backbone_screen", "campaign": str(root), "config": cfg,
        "contextbench_lock": str(lock_path), "contextbench_lock_sha256": sha(lock_path),
        "venv": "/fixture/env", "environment": {"freeze_sha256": "unchanged"}}
    manifest = root/"manifest.json"
    save(manifest, source)
    save(root/"config.json", cfg)
    save(root/"export-audit.json", audit)
    save(root/"export/compression.json", {"ranks": {}, "whole_keep": 1, "dtype": "bfloat16",
        "inference_runtime": INFERENCE_RUNTIME, "weight_sha256": audit["weight_sha256"]})
    save(root/"contextbench/summary.json", summary)
    for ep in summary["episodes"]:
        save(root/"contextbench"/(ep["mode"]+"--"+ep["episode"]+".json"), ep)
    report = screen_report(cfg, lock, audit, summary)
    report["manifest_sha256"] = sha(manifest)
    save(root/"screen-report.json", report)
    return manifest, source


def test_source_zero_is_valid_but_mutation_and_weight_change_are_not(tmp_path):
    manifest, source = source_fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    validate_source(manifest, verify_weights=True)
    spec = {"source_manifest": str(manifest), "config": source["config"],
        "source_files": evidence_files(manifest, source), "diagnostic_protocol": protocol_identity(FORMAT_ID)}
    verify_inputs(spec)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    path = Path(source["campaign"])/"export/tokenizer_config.json"
    save(path, {"added_file": True})
    with pytest.raises(ValueError, match="changed"):
        verify_inputs(spec)
    path.unlink()
    (Path(source["campaign"])/"export/model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights changed"):
        verify_inputs(spec, verify_weights=True)


def load_driver(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"compression"))
    spec = importlib.util.spec_from_file_location("format_driver", ROOT/"compression/protocol_screen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_snapshots_source_and_uses_three_correctly_dependent_jobs(tmp_path, monkeypatch, git_repo):
    driver = load_driver(monkeypatch)
    manifest, source = source_fixture(tmp_path)
    monkeypatch.setattr(driver, "ROOT", git_repo)
    monkeypatch.setattr(sys.modules["submit"], "ROOT", git_repo)
    for key, value in {"TINY_CLM_WORK": str(tmp_path), "TINY_CLM_SNAPSHOTS": str(tmp_path/"snapshots"),
        "TINY_CLM_VENV": "/fixture/env", "TINY_CLM_PARTITION": "gpu-fixture", "TINY_CLM_CPU_PARTITION": "cpu-fixture",
        "TINY_CLM_GPU_CONSTRAINT": "fixture", "TINY_CLM_GPUS": "1", "SLURM_JOB_ID": "parent", "SBATCH_GRES": "gpu:1"}.items():
        monkeypatch.setenv(key, value)
    calls, verified = [], []
    def verify(venv, expected):
        verified.append(venv)
        return expected
    monkeypatch.setattr(driver, "verify_environment", verify)
    original = subprocess.check_output
    def scheduler(command, **kwargs):
        if command[0] != "sbatch":
            return original(command, **kwargs)
        assert verified == [source["venv"]]
        assert "SLURM_JOB_ID" not in kwargs["env"] and "SBATCH_GRES" not in kwargs["env"]
        calls.append(command)
        return str(200+len(calls))+"\n"
    monkeypatch.setattr(subprocess, "check_output", scheduler)
    campaign = tmp_path/"format-test"
    args = ["--source-manifest", str(manifest), "--campaign", str(campaign)]
    driver.main(args)
    assert not campaign.exists() and not calls and not verified
    source_before = {p: p.read_bytes() for p in Path(source["campaign"]).rglob("*") if p.is_file()}
    driver.main(args+["--submit"])
    spec = json.loads((campaign/"manifest.json").read_text())
    verify_snapshot(spec["snapshot"], spec["snapshot_hash"])
    verify_inputs(spec, verify_weights=True)
    assert spec["config"] == source["config"]
    assert (campaign/"reminder.txt").read_text() == FORMAT_TEXT
    assert len(calls) == 3
    assert "--dependency=afterok:201" in calls[1]
    assert "--dependency=afterany:202" in calls[2]
    assert "--gpus=1" in calls[1] and "--mem=64G" in calls[1]
    assert not any(c.startswith("--gpus=") for cmd in (calls[0], calls[2]) for c in cmd)
    assert not any("--array=" in c for cmd in calls for c in cmd)
    assert source_before == {p: p.read_bytes() for p in Path(source["campaign"]).rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        driver.main(args+["--submit"])


@pytest.mark.parametrize("protocol,missing_application", [(None, False), (FORMAT_ID, False), (FORMAT_ID, True),
                                                        (TOOL_FORMAT_ID, False), (TOOL_FORMAT_ID, True)])
def test_contextbench_routes_agents_preserves_settings_and_records_applied_protocol(tmp_path, monkeypatch, protocol, missing_application):
    from tiny_clm import contextbench as cb
    cfg, _, lock, _, _ = fixture()
    lock.update(source=str(tmp_path/"pinned-source"), budget_tokenizer_cache={"path": str(tmp_path/"tokens")})
    lock_path = tmp_path/"lock.json"
    save(lock_path, lock)
    artifact = tmp_path/"export"
    save(artifact/"compression.json", {"weight_sha256": "unchanged", "inference_runtime": INFERENCE_RUNTIME})
    out = tmp_path/"episodes"
    monkeypatch.setattr(cb, "verify_lock", lambda *args: None)
    monkeypatch.setattr(cb, "tree_hash", lambda path: "selected")
    monkeypatch.setattr(cb.shutil, "which", lambda name: "/fixture/"+name)
    monkeypatch.setattr(cb, "wait_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(cb, "metadata", lambda: {})
    calls = []
    class Server:
        def terminate(self):
            pass
        def wait(self, timeout):
            pass
    monkeypatch.setattr(cb.subprocess, "Popen", lambda *args, **kwargs: Server())
    def run(command, **kwargs):
        if "trials" not in command:
            return SimpleNamespace(returncode=0)
        calls.append(command)
        name = command[command.index("--trial-name")+1]
        save(out/"trials"/name/"result.json", {"verifier_result": {"rewards": {"grader_valid": 1, "reward": 0}}})
        if protocol and not missing_application:
            applied = application()
            applied["diagnostic_protocol"] = protocol_identity(protocol)
            save(out/"trials"/name/"agent/answer_format_protocol.json", applied)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cb.subprocess, "run", run)
    args = (artifact, cfg, lock_path, out)
    kwargs = dict(task_names=cfg["screen"]["task_names"], diagnostic_protocol=protocol)
    if missing_application:
        with pytest.raises(RuntimeError, match="failed episodes"):
            cb.run(*args, **kwargs)
    else:
        cb.run(*args, **kwargs)
    summary = json.loads((out/"summary.json").read_text())
    assert summary["all_complete"] is not missing_application
    assert len(calls) == 2
    for mode, command, ep in zip(cfg["contextbench"]["modes"], calls, summary["episodes"]):
        assert command[command.index("-a")+1] == agent_path(mode, protocol)
        settings = [command[i+1] for i, v in enumerate(command) if v == "--agent-kwarg"]
        for key in ("max_tokens", "max_steps", "temperature", "top_p"):
            assert key+"="+json.dumps(cfg["contextbench"][key]) in settings
        assert ep["weight_sha256"] == "unchanged"
        assert ep.get("diagnostic_protocol") == protocol_identity(protocol)
        if missing_application:
            assert ep["error"]["type"] == "ProtocolApplicationError"
    opposite = None if protocol else FORMAT_ID
    with pytest.raises(ValueError, match="another experiment"):
        cb.run(*args, task_names=cfg["screen"]["task_names"], diagnostic_protocol=opposite)
