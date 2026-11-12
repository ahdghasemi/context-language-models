import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_dependencies_and_unique_grid():
    submit = load_script("submit")
    config = json.loads((ROOT/"configs/pilot.json").read_text())
    rows = submit.grid(config, "pilot")
    assert len({r["run_id"] for r in rows}) == len(rows) == 10
    plans = {p[0]: p for p in submit.make_plan("pilot", len(rows))}
    assert plans["train"][3] == [("afterok", "smoke")]
    assert plans["export"][3] == [("aftercorr", "train")]
    assert plans["evaluate"][3] == [("aftercorr", "export")]
    assert plans["profile"][3] == [("aftercorr", "export")]
    assert plans["contextbench"][3] == [("aftercorr", "export")]
    assert all(d[0] == "afterany" for d in plans["aggregate"][3])
    assert submit.make_plan("prepare", 0) == [("prepare", False, None, [])]


def test_aggregate_keeps_missing_runs_and_gate_fails(tmp_path):
    module = load_script("aggregate")
    config = json.loads((ROOT/"configs/pilot.json").read_text())
    spec = {"campaign": str(tmp_path), "config": config, "commit": "test",
            "rows": [{"run_id": "absent", "config": config}]}
    rows, episodes = module.collect(spec)
    assert len(rows) == 1 and rows[0]["status"] == "incomplete"
    assert "evaluation.json" in rows[0]["missing"]
    assert not module.gate(spec, rows, episodes)["passed"]


def test_single_gpu_resource_contract(monkeypatch):
    submit = load_script("submit")
    monkeypatch.setenv("TINY_CLM_GPUS", "2")
    with pytest.raises(ValueError):
        submit.resource_args(True, "train")


def test_baseline_stage_and_gate_protocol():
    import copy
    from tiny_clm.provenance import protocol_contract, require_gate
    submit = load_script("submit")
    config = json.loads((ROOT/"configs/pilot.json").read_text())
    assert [r["config"]["method"] for r in submit.grid(config, "baseline")] == ["original"]
    gate = {"passed": True, "gate_type": "pilot", "protocol_contract": protocol_contract(config)}
    main = json.loads((ROOT/"configs/main.json").read_text())
    require_gate(gate, main, "pilot")
    for field, value in [("revision", "0"*40), ("id", "other")]:
        changed = copy.deepcopy(main)
        changed["model"][field] = value
        with pytest.raises(ValueError, match="protocol"):
            require_gate(gate, changed, "pilot")
    changed = copy.deepcopy(main)
    changed["training"]["optimizer_policy"] = "flow_reset"
    with pytest.raises(ValueError, match="protocol"):
        require_gate(gate, changed, "pilot")
    with pytest.raises(ValueError):
        require_gate(gate, main, "baseline")


def test_empty_clm_results_cannot_pass(tmp_path):
    aggregate = load_script("aggregate")
    config = json.loads((ROOT/"configs/pilot.json").read_text())
    summary = {"contextbench": config["contextbench"], "source_revision": config["contextbench"]["revision"],
               "episodes": [], "all_complete": True}
    with pytest.raises(ValueError, match="coverage"):
        aggregate.validate_episodes(config, summary)


def test_legacy_precision_cannot_pass_current_baseline_gate():
    from tiny_clm.provenance import INFERENCE_RUNTIME, require_gate, protocol_contract
    aggregate = load_script("aggregate")
    config = json.loads((ROOT/"configs/pilot.json").read_text())
    spec = {"stage": "baseline", "config": config, "commit": "fixture"}
    rows = [{"run_id": "original", "status": "complete", "method": "original",
             "inference_runtime": "whole-model-dtype-cast-v1"}]
    episodes = {"original": [{"mode": "clm", "rewards": {"reward": 1.0}}]}
    assert not aggregate.gate(spec, rows, episodes)["passed"]
    rows[0]["inference_runtime"] = INFERENCE_RUNTIME
    assert aggregate.gate(spec, rows, episodes)["passed"]
    contract = protocol_contract(config, baseline=True)
    del contract["inference_runtime"]
    with pytest.raises(ValueError, match="protocol"):
        require_gate({"passed": True, "gate_type": "baseline", "protocol_contract": contract}, config, "baseline")


def test_mixed_episode_precision_is_rejected():
    from tiny_clm.provenance import INFERENCE_RUNTIME
    aggregate = load_script("aggregate")
    config = json.loads((ROOT/"configs/pilot.json").read_text())
    summary = {"contextbench": config["contextbench"], "source_revision": config["contextbench"]["revision"],
               "inference_runtime": INFERENCE_RUNTIME, "episodes": [{"inference_runtime": "legacy"}]}
    with pytest.raises(ValueError, match="mixed inference precision"):
        aggregate.validate_episodes(config, summary)


def test_snapshot_detects_overwritten_and_added_source(tmp_path):
    from tiny_clm.provenance import snapshot_files, verify_snapshot, canonical_hash
    (tmp_path/"code.py").write_text("original")
    record = snapshot_files(tmp_path)
    (tmp_path/"SNAPSHOT_FILES.json").write_text(json.dumps(record))
    assert verify_snapshot(tmp_path, canonical_hash(record)) == canonical_hash(record)
    (tmp_path/"code.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(tmp_path)
    (tmp_path/"code.py").write_text("original")
    (tmp_path/"extra.py").write_text("injected")
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(tmp_path)



def test_small_job_resources_are_configurable_without_duplicate_flags(monkeypatch):
    submit = load_script("submit")
    monkeypatch.setenv("TINY_CLM_SMOKE_MEM", "12G")
    monkeypatch.setenv("TINY_CLM_SMOKE_CPUS", "3")
    args = submit.resource_args(False, "smoke")
    assert [x for x in args if x.startswith("--mem=")] == ["--mem=12G"]
    assert [x for x in args if x.startswith("--cpus-per-task=")] == ["--cpus-per-task=3"]


def test_real_git_snapshot_then_mock_scheduler_dependencies(tmp_path, monkeypatch):
    import subprocess
    from tiny_clm.provenance import verify_snapshot
    submit = load_script("submit")
    repo = tmp_path/"repo"
    repo.mkdir()
    (repo/"file.txt").write_text("committed source")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "file.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    monkeypatch.setattr(submit, "ROOT", repo)
    venv = tmp_path/"env"
    venv.mkdir()
    (venv/"READY.json").write_text(json.dumps({"freeze_sha256": "fixture"}))
    verified = []

    def verify_fixture(path):
        verified.append(path)
        return json.loads((Path(path)/"READY.json").read_text())

    monkeypatch.setattr(submit, "verify_environment", verify_fixture)
    monkeypatch.setenv("TINY_CLM_VENV", str(venv))
    monkeypatch.setenv("TINY_CLM_SNAPSHOTS", str(tmp_path/"snapshots"))
    monkeypatch.setenv("TINY_CLM_GPU_CONSTRAINT", "test-fixture-not-a-cluster-setting")
    monkeypatch.setenv("SLURM_JOB_ID", "parent-job")
    monkeypatch.setenv("SLURM_GPUS_PER_NODE", "a100:1")
    monkeypatch.setenv("SBATCH_GRES", "gpu:a100:1")
    config = ROOT/"configs/pilot.json"
    original = subprocess.check_output
    calls = []

    def mocked_submit(command, **kw):
        if command[0] == "sbatch":
            assert verified == [venv]
            assert "--export=ALL" in command
            assert not any(k in kw["env"] for k in
                           ("SLURM_JOB_ID", "SLURM_GPUS_PER_NODE", "SBATCH_GRES"))
            assert kw["env"]["TINY_CLM_VENV"] == str(venv)
            calls.append(command)
            return str(100+len(calls))+"\n"
        return original(command, **kw)

    monkeypatch.setattr(submit.subprocess, "check_output", mocked_submit)
    campaign = tmp_path/"campaign"
    submit.main(["correctness", "--config", str(config), "--campaign", str(campaign), "--submit"])
    manifest = json.loads((campaign/"manifest.json").read_text())
    verify_snapshot(manifest["snapshot"], manifest["snapshot_hash"])
    assert "SLURM_GPUS_PER_NODE" in manifest["submission_environment_removed"]
    assert len(calls) == 2 and "--dependency=afterok:101" in calls[1]
    assert json.loads((campaign/"jobs.json").read_text()) == {"correctness": "101", "smoke": "102"}

    def reject_environment(path):
        raise RuntimeError("fixture installed environment mismatch")

    monkeypatch.setattr(submit, "verify_environment", reject_environment)
    blocked = tmp_path/"blocked-campaign"
    with pytest.raises(RuntimeError, match="installed environment mismatch"):
        submit.main(["correctness", "--config", str(config), "--campaign", str(blocked), "--submit"])
    assert not blocked.exists()
    assert len(calls) == 2  # No additional job was submitted on mismatch.


def test_resume_filters_parent_resources_and_keeps_resume_export(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.syspath_prepend(str(ROOT))
    resume = load_script("resume")
    run = tmp_path/"runs"/"fixture"
    run.mkdir(parents=True)
    (run/"status.json").write_text(json.dumps({"status": "incomplete"}))
    (run/"latest.pt").touch()
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"campaign": str(tmp_path), "snapshot": "/snapshot",
                                    "rows": [{"run_id": "fixture"}],
                                    "venv": "/fixture-env", "environment": {"fixture": True}}))
    (tmp_path/"jobs.json").write_text(json.dumps({"train": "100"}))
    monkeypatch.setenv("USER", "fixture")
    monkeypatch.setenv("SLURM_GPUS_PER_NODE", "a100:1")
    monkeypatch.setenv("TINY_CLM_GPUS", "1")
    calls = []
    verified = []
    monkeypatch.setattr(resume, "verify_environment", lambda path, expected:
                        verified.append((path, expected)))

    def scheduler(command, **kwargs):
        if command[0] == "squeue":
            return ""
        assert command[0] == "sbatch"
        assert verified == [("/fixture-env", {"fixture": True})]
        assert "SLURM_GPUS_PER_NODE" not in kwargs["env"]
        calls.append(command)
        return str(200+len(calls))+"\n"

    monkeypatch.setattr(subprocess, "check_output", scheduler)
    monkeypatch.setattr("sys.argv", ["resume.py", "--manifest", str(manifest),
                                    "--indices", "0", "--from-stage", "train", "--submit"])
    resume.main()
    assert len(calls) == 5
    assert [x for x in calls[0] if x.startswith("--export=")] == ["--export=ALL,TINY_CLM_RESUME=1"]
    assert all([x for x in cmd if x.startswith("--export=")] == ["--export=ALL"]
               for cmd in calls[1:])
    assert "--dependency=aftercorr:201" in calls[1]
