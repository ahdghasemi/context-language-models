"""Separate development/validation evidence and bounded resource-probe execution."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tiny_clm.answer_format import FORMAT_ID, FORMAT_TEXT, protocol_identity
from tiny_clm.protocol_screen import compare_report, evidence_files as original_files, sha
from tiny_clm.protocol_validation import (development, expanded_config, evidence_files, verify_inputs, validation_report)
from tiny_clm.provenance import require_gate, verify_snapshot
from tiny_clm.training_memory import estimates, probe_config, probe
from tiny_clm.train import train, validate
from tiny_clm.smoke import prepare_tiny
from test_format_screen import source_fixture, variant_summary, save
from test_screen import git_repo

ROOT = Path(__file__).resolve().parents[2]


def development_fixture(tmp_path):
    manifest, source = source_fixture(tmp_path)
    lock_path = Path(source["contextbench_lock"])
    lock = json.loads(lock_path.read_text())
    lock["tasks"] += [{"path": "/task/kv-store-"+level+"-seed1102", "sha256": level}
                      for level in ("R130", "R1000")]
    save(lock_path, lock)
    source["contextbench_lock_sha256"] = sha(lock_path)
    save(manifest, source)
    original_root = Path(source["campaign"])
    original = json.loads((original_root/"screen-report.json").read_text())
    original["manifest_sha256"] = sha(manifest)
    save(original_root/"screen-report.json", original)
    summary = json.loads((original_root/"contextbench/summary.json").read_text())
    audit = json.loads((original_root/"export-audit.json").read_text())
    root = tmp_path/"development"
    spec = {"stage": "protocol_screen", "campaign": str(root), "config": source["config"],
        "source_manifest": str(manifest), "source_files": original_files(manifest, source),
        "diagnostic_protocol": protocol_identity(FORMAT_ID), "venv": source["venv"], "environment": source["environment"]}
    dev_manifest = root/"manifest.json"
    save(dev_manifest, spec)
    save(root/"config.json", spec["config"])
    (root/"reminder.txt").write_text(FORMAT_TEXT)
    save(root/"checks.json", {"passed": True})
    variant = variant_summary(summary)
    save(root/"contextbench/summary.json", variant)
    for ep in variant["episodes"]:
        save(root/"contextbench"/(ep["mode"]+"--"+ep["episode"]+".json"), ep)
    report = compare_report(spec["config"], lock, audit, summary, variant)
    report["manifest_sha256"] = sha(dev_manifest)
    save(root/"protocol-report.json", report)
    return dev_manifest, spec, lock, audit, variant


def validation_fixture(tmp_path):
    manifest, dev, lock, audit, variant = development_fixture(tmp_path)
    cfg = expanded_config(dev, lock)
    spec = {"config": cfg, "development_manifest": str(manifest), "source_files": evidence_files(manifest, dev),
        "diagnostic_protocol": protocol_identity(FORMAT_ID), "memory_config": probe_config(cfg), "manifest_sha256": "test"}
    summary = copy.deepcopy(variant)
    summary["screen_task_names"] = cfg["screen"]["task_names"]
    summary["episodes"] = []
    for task in lock["tasks"]:
        for template in variant["episodes"]:
            ep = copy.deepcopy(template)
            ep.update(episode=Path(task["path"]).name, task_sha256=task["sha256"])
            summary["episodes"].append(ep)
    memory = {"model": cfg["model"], "training_config": spec["memory_config"], "sequence_length": 512,
        "weight_sha256": audit["weight_sha256"], "manifest_sha256": "test", "probe_complete": True, "passed": True,
        "memory": {"cuda_peak_allocated": 70*2**30, "cuda_peak_reserved": 72*2**30},
        "device_total_bytes": 80*2**30, "reserved_headroom_bytes": 8*2**30,
        "trainer_status": {"total_steps": 2, "training_tokens": 4088},
        "budget": {"original_parameters": estimates(cfg)["parameters"],
                   "deployment_parameters": estimates(cfg)["predicted_deployment_parameters"]}}
    return spec, dev, lock, audit, summary, memory


def test_full_coverage_separate_development_and_no_automatic_compression(tmp_path):
    spec, dev, lock, audit, summary, memory = validation_fixture(tmp_path)
    verify_inputs(spec, verify_weights=True)
    report = validation_report(spec, dev, lock, audit, summary, memory)
    assert report["passed"] and report["complete"] and not report["can_advance_to_compression"]
    assert report["gate_type"] == "protocol_validation"
    assert len(report["cohorts"]["clm/development_repeat"]["rewards"]) == 1
    assert len(report["cohorts"]["clm/additional_validation"]["rewards"]) == 3
    for kind in ("baseline", "pilot"):
        with pytest.raises(ValueError):
            require_gate(report, spec["config"], kind)


@pytest.mark.parametrize("fault", ["episode", "protocol", "memory", "oom", "quality", "cap", "model", "steps", "measured_steps", "headroom"])
def test_missing_mixed_failed_or_negative_results_cannot_pass(tmp_path, fault):
    spec, dev, lock, audit, summary, memory = validation_fixture(tmp_path)
    if fault == "episode":
        summary["episodes"].pop()
    elif fault == "protocol":
        summary["episodes"][0].pop("protocol_application")
    elif fault == "memory":
        memory = None
    elif fault == "oom":
        memory.update(probe_complete=False, passed=False, error="out of memory")
    elif fault == "quality":
        summary["episodes"][0]["rewards"]["reward"] = 0
    elif fault == "cap":
        summary["contextbench"]["max_tokens"] += 1
    elif fault == "model":
        memory["weight_sha256"] = "wrong"
    elif fault == "steps":
        memory["training_config"] = copy.deepcopy(memory["training_config"])
        memory["training_config"]["training"]["accumulation"] = 1
    elif fault == "measured_steps":
        memory["trainer_status"]["total_steps"] = 1
    else:
        memory["memory"]["cuda_peak_reserved"] = memory["device_total_bytes"]-2**30
        memory["reserved_headroom_bytes"] = 2**30
    report = validation_report(spec, dev, lock, audit, summary, memory)
    assert not report["passed"]
    assert report["complete"] is (fault in {"quality", "headroom"})


def test_development_report_and_frozen_source_cannot_be_replaced(tmp_path):
    spec, dev, _, _, _, _ = validation_fixture(tmp_path)
    verify_inputs(spec)
    path = Path(dev["campaign"])/"protocol-report.json"
    report = json.loads(path.read_text())
    report["comparison"][0]["reminder_reward"] = 0.5
    save(path, report)
    with pytest.raises(ValueError, match="consistent"):
        verify_inputs(spec)


def test_memory_policy_size_and_unsupported_cpu_failure_are_explicit(tmp_path):
    spec, _, _, _, _, _ = validation_fixture(tmp_path)
    cfg = spec["config"]
    estimate = estimates(cfg)
    assert estimate["all_mlp_subtotal_bytes"] == 85876269056
    assert estimate["down_subtotal_bytes"] == 71531749376
    assert estimate["down_rank"] == 1216
    assert 0.089 < estimate["predicted_whole_reduction"] < 0.09
    assert estimate["down_subtotal_bytes"]+8*2**30 < 85094825984
    assert spec["memory_config"]["training"]["adam_foreach"] is False
    if not torch.cuda.is_available():
        result = probe(tmp_path/"unused", cfg, tmp_path/"probe.json")
        assert not result["passed"] and "CUDA" in result["error"]["message"]
        assert (tmp_path/"probe.json").is_file()


@pytest.mark.parametrize("foreach", [False, True])
def test_adam_execution_option_preserves_full_training_and_resume(tmp_path, foreach):
    model, cfg, data = prepare_tiny(tmp_path/"assets")
    initial = copy.deepcopy(model)
    cfg["training"]["adam_foreach"] = foreach
    assert train(model, data["train"], data["validation"], cfg, tmp_path/"full")
    assert not train(initial, data["train"], data["validation"], cfg, tmp_path/"resume", stop_after=2)
    assert train(initial, data["train"], data["validation"], cfg, tmp_path/"resume", resume=True)
    for p, q in zip(model.parameters(), initial.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    checkpoint = torch.load(tmp_path/"resume/latest.pt", weights_only=False)
    assert checkpoint["config"]["training"]["adam_foreach"] is foreach
    cfg["training"]["adam_foreach"] = "false"
    with pytest.raises(ValueError, match="adam_foreach"):
        validate(cfg)


def test_adam_single_tensor_and_foreach_match_on_cpu():
    torch.manual_seed(4)
    p = torch.nn.Parameter(torch.randn(7, 5))
    q = torch.nn.Parameter(p.detach().clone())
    optimizers = [torch.optim.Adam([p], lr=1e-3, foreach=False), torch.optim.Adam([q], lr=1e-3, foreach=True)]
    for _ in range(4):
        grad = torch.randn_like(p)
        for param, opt in zip((p, q), optimizers):
            param.grad = grad.clone()
            opt.step()
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)


def test_dry_run_and_submission_freeze_inputs_serialize_gpu_jobs_and_preserve_sources(tmp_path, monkeypatch, git_repo):
    monkeypatch.syspath_prepend(str(ROOT/"compression"))
    loader = importlib.util.spec_from_file_location("validation_driver", ROOT/"compression/validate_protocol.py")
    driver = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(driver)
    manifest, dev, _, _, _ = development_fixture(tmp_path)
    monkeypatch.setattr(driver, "ROOT", git_repo)
    monkeypatch.setattr(sys.modules["submit"], "ROOT", git_repo)
    for k, v in {"TINY_CLM_WORK": str(tmp_path), "TINY_CLM_SNAPSHOTS": str(tmp_path/"snapshots"),
        "TINY_CLM_VENV": "/fixture/env", "TINY_CLM_PARTITION": "gpu-fixture", "TINY_CLM_CPU_PARTITION": "cpu-fixture",
        "TINY_CLM_GPU_CONSTRAINT": "fixture", "TINY_CLM_GPUS": "1", "SLURM_JOB_ID": "parent", "SBATCH_GRES": "gpu:1"}.items():
        monkeypatch.setenv(k, v)
    checked, calls = [], []
    def verify(venv, expected):
        checked.append(venv)
        return expected
    monkeypatch.setattr(driver, "verify_environment", verify)
    original = subprocess.check_output
    def scheduler(command, **kwargs):
        if command[0] != "sbatch":
            return original(command, **kwargs)
        assert checked == [dev["venv"]]
        assert "SLURM_JOB_ID" not in kwargs["env"] and "SBATCH_GRES" not in kwargs["env"]
        calls.append(command)
        return str(300+len(calls))+"\n"
    monkeypatch.setattr(subprocess, "check_output", scheduler)
    campaign = tmp_path/"validation"
    args = ["--source-manifest", str(manifest), "--campaign", str(campaign)]
    driver.main(args)
    assert not campaign.exists() and not calls and not checked
    source_before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    driver.main(args+["--submit"])
    spec = json.loads((campaign/"manifest.json").read_text())
    verify_snapshot(spec["snapshot"], spec["snapshot_hash"])
    verify_inputs(spec, verify_weights=True)
    assert len(calls) == 4
    assert "--dependency=afterok:301" in calls[1]
    assert "--dependency=afterok:301,afterany:302" in calls[2]
    assert "--dependency=afterany:302,afterany:303" in calls[3]
    assert "--mem=128G" in calls[2] and "--gpus=1" in calls[2]
    assert all(p.read_bytes() == value for p, value in source_before.items())
    with pytest.raises(FileExistsError):
        driver.main(args+["--submit"])
