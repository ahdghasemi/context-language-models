"""Screen scope, negative results, accounting and real-process submission boundaries."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tiny_clm.provenance import canonical_hash, require_gate, select_task_records, verify_snapshot, INFERENCE_RUNTIME
from tiny_clm.screen_contract import qwen3_counts, screen_report, validate_screen

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    config = json.loads((ROOT/"compression/configs/screen-qwen3-4b.json").read_text())
    reference = {"config": json.loads((ROOT/"compression/configs/pilot.json").read_text())}
    lock = {"runnable": True, "request_hash": canonical_hash(config["contextbench"]),
            "revision": config["contextbench"]["revision"], "image_sha256": "image",
            "tasks": [{"path": "/task/kv-store-R1000-seed1101", "sha256": "unselected"},
                      {"path": "/task/kv-store-R130-seed1101", "sha256": "selected"}]}
    audit = {"reload_passed": True, "cache_passed": True, "inference_runtime": INFERENCE_RUNTIME,
             "model": config["model"], "weight_sha256": "weights"}
    episodes = [{"mode": m, "episode": config["screen"]["task_names"][0],
                 "status": "complete", "error": None, "returncode": 0,
                 "rewards": {"grader_valid": 1, "reward": 0.0}, "weight_sha256": "weights",
                 "task_sha256": "selected", "request_hash": lock["request_hash"], "inference_runtime": INFERENCE_RUNTIME}
                for m in config["contextbench"]["modes"]]
    summary = {"all_complete": True, "inference_runtime": INFERENCE_RUNTIME, "contextbench": config["contextbench"],
               "screen_task_names": config["screen"]["task_names"], "container_sha256": "image",
               "source_revision": lock["revision"], "episodes": episodes}
    return config, reference, lock, audit, summary


def test_subset_preserves_locked_order_and_inputs():
    cfg, ref, lock, _, _ = fixture()
    original = copy.deepcopy((cfg, ref, lock))
    assert validate_screen(cfg, ref, lock) == [lock["tasks"][1]]
    assert select_task_records(lock["tasks"]) == lock["tasks"]
    names = [Path(t["path"]).name for t in reversed(lock["tasks"])]
    assert select_task_records(lock["tasks"], names) == lock["tasks"]
    assert (cfg, ref, lock) == original
    for invalid in ([], ["missing"], [names[0], names[0]]):
        with pytest.raises(ValueError):
            select_task_records(lock["tasks"], invalid)


def test_reject_changed_protocol_training_and_mutable_revision():
    cfg, ref, lock, _, _ = fixture()
    for edit in (lambda c: c["contextbench"].update(max_steps=16),
                 lambda c: c.update(training={}), lambda c: c["model"].update(revision="main")):
        changed = copy.deepcopy(cfg)
        edit(changed)
        with pytest.raises(ValueError):
            validate_screen(changed, ref, lock)


def test_valid_zero_is_complete_but_fails_quality_screen():
    cfg, _, lock, audit, summary = fixture()
    report = screen_report(cfg, lock, audit, summary)
    assert report["complete"] and not report["passed"] and not report["errors"]
    assert len(report["episodes"]) == 2


def test_passing_screen_cannot_authorize_compression():
    cfg, ref, lock, audit, summary = fixture()
    summary["episodes"][0]["rewards"]["reward"] = 1.0
    report = screen_report(cfg, lock, audit, summary)
    assert report["passed"] and report["complete"]
    assert report["can_advance_to_compression"] is False
    for kind in ("baseline", "pilot"):
        with pytest.raises(ValueError, match="passing"):
            require_gate(report, ref["config"], kind)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "weight", "task", "request", "nan", "failed", "container", "runtime", "cache"])
def test_partial_or_mixed_evidence_cannot_pass(mutation):
    cfg, _, lock, audit, summary = fixture()
    summary["episodes"][0]["rewards"]["reward"] = 1.0
    ep = summary["episodes"][0]
    if mutation == "missing":
        summary["episodes"].pop()
    elif mutation == "duplicate":
        summary["episodes"].append(ep)
    elif mutation in {"weight", "task", "request"}:
        ep[{"weight": "weight_sha256", "task": "task_sha256", "request": "request_hash"}[mutation]] = "changed"
    elif mutation == "nan":
        ep["rewards"]["reward"] = float("nan")
    elif mutation == "failed":
        ep["returncode"] = 1
    elif mutation == "runtime":
        ep["inference_runtime"] = "legacy"
    elif mutation == "cache":
        audit["cache_passed"] = False
    else:
        summary["container_sha256"] = "changed"
    report = screen_report(cfg, lock, audit, summary)
    assert not report["complete"] and not report["passed"] and report["errors"]


def test_missing_export_or_summary_is_reported():
    cfg, _, lock, audit, summary = fixture()
    for a, s in [(None, summary), (audit, None), (None, None)]:
        report = screen_report(cfg, lock, a, s)
        assert not report["complete"] and not report["passed"]


def test_audited_parameter_counts_and_large_training_footprint():
    cfg, _, _, _, _ = fixture()
    counts = qwen3_counts(cfg["screen"]["audited_model_config"])
    assert counts["parameters"] == 4022468096
    assert counts["all_mlp_weights"] == 2689597440
    assert counts["bf16_parameter_bytes"] == 8044936192
    assert counts["training_subtotal_excluding_workspaces_bytes"] > 79.9*2**30


def load_driver(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"compression"))
    spec = importlib.util.spec_from_file_location("screen_driver", ROOT/"compression/screen.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


@pytest.fixture
def git_repo(tmp_path):
    # Slurm runs a git archive without .git. Submission tests must supply their
    # own checkout rather than discovering a repository above the test sources.
    repo = tmp_path/"repo"
    repo.mkdir()
    (repo/"source.py").write_text("# committed fixture\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    return repo


def test_real_snapshot_mock_scheduler_and_dry_run(tmp_path, monkeypatch, git_repo):
    driver = load_driver(monkeypatch)
    cfg, ref, lock, _, _ = fixture()
    repo = git_repo
    monkeypatch.setattr(driver, "ROOT", repo)
    monkeypatch.setattr(sys.modules["submit"], "ROOT", repo)
    ref.update(contextbench_dir=str(tmp_path), venv="/fixture/env", environment={"freeze_sha256": "unchanged"})
    for path, value in [("config.json", cfg), ("reference.json", ref), ("contextbench-lock.json", lock)]:
        (tmp_path/path).write_text(json.dumps(value))
    campaign = tmp_path/"campaign"
    args = ["--config", str(tmp_path/"config.json"), "--reference-manifest", str(tmp_path/"reference.json"),
            "--campaign", str(campaign)]
    for key, value in {"TINY_CLM_WORK": str(tmp_path), "TINY_CLM_SNAPSHOTS": str(tmp_path/"snapshots"),
                       "TINY_CLM_VENV": "/fixture/env", "TINY_CLM_PARTITION": "gpu-fixture",
                       "TINY_CLM_CPU_PARTITION": "cpu-fixture", "TINY_CLM_GPU_CONSTRAINT": "fixture",
                       "TINY_CLM_GPUS": "1", "SLURM_JOB_ID": "parent", "SBATCH_GRES": "gpu:1"}.items():
        monkeypatch.setenv(key, value)
    verified, calls = [], []
    def verify(path, expected):
        verified.append((path, expected))
        return expected
    monkeypatch.setattr(driver, "verify_environment", verify)
    original = subprocess.check_output
    def scheduler(command, **kwargs):
        if command[0] != "sbatch":
            return original(command, **kwargs)
        assert verified == [(ref["venv"], ref["environment"])]
        assert "SLURM_JOB_ID" not in kwargs["env"] and "SBATCH_GRES" not in kwargs["env"]
        assert kwargs["env"]["TINY_CLM_VENV"] == ref["venv"]
        calls.append(command)
        return str(100+len(calls))+"\n"
    monkeypatch.setattr(subprocess, "check_output", scheduler)
    driver.main(args)
    assert not campaign.exists() and not calls and not verified
    driver.main(args+["--submit"])
    manifest = json.loads((campaign/"manifest.json").read_text())
    verify_snapshot(manifest["snapshot"], manifest["snapshot_hash"])
    assert len(calls) == 5
    for index, parent in [(1, 101), (2, 102), (3, 103)]:
        assert f"--dependency=afterok:{parent}" in calls[index]
    assert "--dependency=afterany:104" in calls[4]
    for index in [0, 1, 4]:
        assert not any(c.startswith("--gpus=") for c in calls[index])
    for index in [2, 3]:
        assert "--gpus=1" in calls[index] and "--mem=64G" in calls[index]
    assert not any("--array=" in c for cmd in calls for c in cmd)
    assert set(json.loads((campaign/"jobs.json").read_text())) == {p[0] for p in driver.PLAN}
    with pytest.raises(FileExistsError):
        driver.main(args+["--submit"])


def test_environment_failure_blocks_all_submissions(tmp_path, monkeypatch):
    driver = load_driver(monkeypatch)
    cfg, ref, lock, _, _ = fixture()
    ref.update(contextbench_dir=str(tmp_path), venv="/fixture/env", environment={})
    for name, value in [("config.json", cfg), ("reference.json", ref), ("contextbench-lock.json", lock)]:
        (tmp_path/name).write_text(json.dumps(value))
    for key in ("TINY_CLM_PARTITION", "TINY_CLM_CPU_PARTITION", "TINY_CLM_GPU_CONSTRAINT"):
        monkeypatch.setenv(key, "fixture")
    monkeypatch.setenv("TINY_CLM_GPUS", "1")
    def git_only(command, **kw):
        assert command[0] == "git"  # Any sbatch would fail this test.
        return "" if "status" in command else "a"*40
    monkeypatch.setattr(subprocess, "check_output", git_only)
    def bad_environment(*_):
        raise RuntimeError("installed environment mismatch")
    monkeypatch.setattr(driver, "verify_environment", bad_environment)
    with pytest.raises(RuntimeError, match="environment mismatch"):
        driver.main(["--config", str(tmp_path/"config.json"), "--reference-manifest", str(tmp_path/"reference.json"),
                     "--campaign", str(tmp_path/"blocked"), "--submit"])
    assert not (tmp_path/"blocked").exists()


def test_prepare_only_dry_run_has_no_gpu_or_inference(tmp_path, monkeypatch, capsys, git_repo):
    driver = load_driver(monkeypatch)
    monkeypatch.setattr(driver, "ROOT", git_repo)
    cfg, ref, lock, _, _ = fixture()
    ref.update(contextbench_dir=str(tmp_path), venv="/fixture/env")
    for name, value in [("config.json", cfg), ("reference.json", ref), ("contextbench-lock.json", lock)]:
        (tmp_path/name).write_text(json.dumps(value))
    driver.main(["--config", str(tmp_path/"config.json"), "--reference-manifest", str(tmp_path/"reference.json"),
                 "--campaign", str(tmp_path/"absent"), "--prepare-only"])
    out = capsys.readouterr().out
    assert "screen_prepare" in out and "screen_check" in out
    assert "screen_export" not in out and "--gpus=" not in out
    commit = subprocess.check_output(["git", "-C", str(git_repo), "rev-parse", "HEAD"], text=True).strip()
    assert json.loads(out[out.index("{\n"):])["commit"] == commit
    assert not (tmp_path/"absent").exists()
