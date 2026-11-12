"""Native tool-call guidance stays a separate prompt-only development experiment."""
import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tiny_clm.answer_format import (FORMAT_ID, FORMAT_TEXT, TOOL_FORMAT_ID, TOOL_FORMAT_TEXT,
    ToolFormatReminderMixin, protocol_identity, protocol_text, add_reminder, agent_path)
from tiny_clm.protocol_screen import validate_reference, evidence_files, verify_inputs, compare_report, TOOL_SCREEN_TASKS, sha
from tiny_clm.protocol_validation import validation_report, development
from tiny_clm.provenance import require_gate, verify_snapshot
from tiny_clm.screen_contract import screen_report
from tiny_clm.serve import parse_response
from test_format_screen import save, load_driver
from test_protocol_validation import validation_fixture
from test_screen import git_repo


def source_validation(tmp_path):
    spec, dev, lock, audit, summary, memory = validation_fixture(tmp_path)
    root = tmp_path/"validation"
    spec.pop("manifest_sha256")
    spec.update(stage="protocol_validation", campaign=str(root), commit="fixture",
                venv=dev["venv"], environment=dev["environment"])
    manifest = root/"manifest.json"
    save(manifest, spec)
    save(root/"config.json", spec["config"])
    (root/"reminder.txt").write_text(FORMAT_TEXT)
    for ep in summary["episodes"]:
        if ep["episode"] != "kv-store-R130-seed1101" or ep["mode"] != "clm":
            ep["rewards"]["reward"] = 0.0
        save(root/"contextbench"/(ep["mode"]+"--"+ep["episode"]+".json"), ep)
    save(root/"contextbench/summary.json", summary)
    memory["manifest_sha256"] = sha(manifest)
    save(root/"memory-probe.json", memory)
    report = validation_report({**spec, "manifest_sha256": sha(manifest)}, dev, lock, audit, summary, memory)
    report["manifest_sha256"] = sha(manifest)
    save(root/"validation-report.json", report)
    save(root/"checks.json", {"passed": True, "manifest_sha256": sha(manifest)})
    assert report["complete"] and not report["passed"]
    return manifest, spec


def new_summary(baseline, cfg):
    variant = copy.deepcopy(baseline)
    variant["screen_task_names"] = cfg["screen"]["task_names"]
    variant["diagnostic_protocol"] = protocol_identity(TOOL_FORMAT_ID)
    variant["episodes"] = [e for e in variant["episodes"] if e["episode"] in TOOL_SCREEN_TASKS]
    for ep in variant["episodes"]:
        ep["diagnostic_protocol"] = protocol_identity(TOOL_FORMAT_ID)
        ep["protocol_application"]["diagnostic_protocol"] = protocol_identity(TOOL_FORMAT_ID)
        ep["rewards"].update(reward=1.0, n_answered=24)
    return variant


def test_only_adds_tool_guidance_preserving_original_answer_bytes_and_delegation(tmp_path):
    received = []
    class Parent:
        async def run(self, instruction, environment, context):
            received.append((instruction, environment, context))
            return context
    class Wrapped(ToolFormatReminderMixin, Parent):
        logs_dir = tmp_path
    original = "# Key-Value Store\nTask.\n"
    env, context = object(), object()
    assert asyncio.run(Wrapped().run(original, env, context)) is context
    assert received == [(original.rstrip()+"\n\n"+FORMAT_TEXT+"\n"+TOOL_FORMAT_TEXT, env, context)]
    assert protocol_identity(FORMAT_ID)["suffix_sha256"] == "9bcff2229e2bebfa27a7490bb603ee8873abcd85ec0486f66aebf11301fb6e80"
    assert protocol_text(TOOL_FORMAT_ID).startswith(FORMAT_TEXT+"\n")
    for mode in ("clm", "append_only"):
        assert "ToolFormatReminder" in agent_path(mode, TOOL_FORMAT_ID)
        assert "ToolFormatReminder" not in agent_path(mode, FORMAT_ID)
    saved = json.loads((tmp_path/"answer_format_protocol.json").read_text())
    assert saved["diagnostic_protocol"] == protocol_identity(TOOL_FORMAT_ID)
    with pytest.raises(ValueError, match="already"):
        add_reminder(received[0][0], TOOL_FORMAT_ID)


@pytest.mark.parametrize("text", ["echo READY_FOR_NEXT_OP", 'bash {"command":"pwd"}',
    '```bash\npwd\n```', '<tool_call>{"name":"bash","arguments":{"command":"unfinished'])
def test_observed_non_calls_stay_unexecuted_by_parser(text):
    # The intervention must not broaden parser acceptance or auto-run prose.
    message, finish = parse_response(text)
    assert finish == "stop" and not message.get("tool_calls")
    assert message["content"] == text


def test_example_uses_the_existing_parser_without_repair():
    start, end = TOOL_FORMAT_TEXT.index("<tool_call>"), TOOL_FORMAT_TEXT.index("</tool_call>")+len("</tool_call>")
    message, finish = parse_response(TOOL_FORMAT_TEXT[start:end])
    assert finish == "tool_calls"
    assert message["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"command": "pwd"}


def test_valid_negative_source_is_read_only_and_protocol_cannot_authorize_training(tmp_path):
    manifest, parent = source_validation(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    source, lock, audit, baseline, cfg = validate_reference(manifest, TOOL_FORMAT_ID, verify_weights=True)
    assert cfg["screen"]["task_names"] == TOOL_SCREEN_TASKS
    assert cfg["contextbench"] == parent["config"]["contextbench"]
    variant = new_summary(baseline, cfg)
    report = compare_report(cfg, lock, audit, baseline, variant, diagnostic_protocol=TOOL_FORMAT_ID)
    assert report["complete"] and report["passed"] and report["gate_type"] == "tool_call_screen"
    assert not report["can_advance_to_compression"] and len(report["episodes"]) == 4
    assert all(row["answer_format_control_reward"] == 0 and row["tool_guidance_reward"] == 1
               and "original_reward" not in row for row in report["comparison"])
    assert not screen_report(cfg, lock, audit, variant, diagnostic_protocol=FORMAT_ID)["complete"]
    spec = {"config": cfg, "source_manifest": str(manifest),
        "source_files": evidence_files(manifest, source, TOOL_FORMAT_ID),
        "diagnostic_protocol": protocol_identity(TOOL_FORMAT_ID)}
    verify_inputs(spec, verify_weights=True)
    changed = copy.deepcopy(spec)
    changed["config"]["contextbench"]["max_tokens"] *= 2
    with pytest.raises(ValueError, match="changed"):
        verify_inputs(changed)
    for kind in ("baseline", "pilot"):
        with pytest.raises(ValueError):
            require_gate(report, cfg, kind)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["report", "episode", "weights", "checks"])
def test_changed_source_evidence_fails(tmp_path, fault):
    manifest, parent = source_validation(tmp_path)
    source, _, _, _, _ = validate_reference(manifest, TOOL_FORMAT_ID)
    root = Path(parent["campaign"])
    if fault == "weights":
        (Path(source["campaign"])/"export/model.safetensors").write_bytes(b"changed")
    else:
        filename = {"report": "validation-report.json", "episode": "contextbench/clm--kv-store-R1000-seed1101.json",
                    "checks": "checks.json"}[fault]
        value = json.loads((root/filename).read_text())
        value["passed" if fault != "episode" else "status"] = "tampered"
        save(root/filename, value)
    with pytest.raises(ValueError):
        validate_reference(manifest, TOOL_FORMAT_ID, verify_weights=True)


def test_tool_protocol_cannot_be_used_as_answer_only_development(tmp_path):
    manifest = tmp_path/"manifest.json"
    save(manifest, {"stage": "protocol_screen", "diagnostic_protocol": protocol_identity(TOOL_FORMAT_ID)})
    with pytest.raises(ValueError, match="format-development"):
        development(manifest)


def test_tool_screen_submission_reuses_export_and_serial_dependencies(tmp_path, monkeypatch, git_repo):
    driver = load_driver(monkeypatch)
    manifest, parent = source_validation(tmp_path)
    monkeypatch.setattr(driver, "ROOT", git_repo)
    monkeypatch.setattr(sys.modules["submit"], "ROOT", git_repo)
    for key, value in {"TINY_CLM_WORK": str(tmp_path), "TINY_CLM_SNAPSHOTS": str(tmp_path/"snapshots"),
        "TINY_CLM_VENV": "/fixture/env", "TINY_CLM_PARTITION": "gpu-fixture", "TINY_CLM_CPU_PARTITION": "cpu-fixture",
        "TINY_CLM_GPU_CONSTRAINT": "fixture", "TINY_CLM_GPUS": "1", "SLURM_JOB_ID": "parent"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(driver, "verify_environment", lambda venv, expected: expected)
    calls = []
    original = subprocess.check_output
    def scheduler(command, **kwargs):
        if command[0] != "sbatch":
            return original(command, **kwargs)
        assert "SLURM_JOB_ID" not in kwargs["env"]
        calls.append(command)
        return str(500+len(calls))+"\n"
    monkeypatch.setattr(subprocess, "check_output", scheduler)
    campaign = tmp_path/"tool-format"
    args = ["--intervention", "tool-call-format", "--source-manifest", str(manifest), "--campaign", str(campaign)]
    driver.main(args)
    assert not campaign.exists() and not calls
    driver.main(args+["--submit"])
    spec = json.loads((campaign/"manifest.json").read_text())
    verify_snapshot(spec["snapshot"], spec["snapshot_hash"])
    verify_inputs(spec, verify_weights=True)
    assert spec["config"]["contextbench"] == parent["config"]["contextbench"]
    assert (campaign/"reminder.txt").read_text() == protocol_text(TOOL_FORMAT_ID)
    assert len(calls) == 3 and "--dependency=afterok:501" in calls[1] and "--dependency=afterany:502" in calls[2]
    assert "--time=03:00:00" in calls[1] and "--mem=64G" in calls[1]
    assert not (campaign/"export").exists()
