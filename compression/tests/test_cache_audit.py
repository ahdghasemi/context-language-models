import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, Qwen3Config

from tiny_clm.cache_audit import capture, comparison, validate_source
from tiny_clm.io import file_sha
from tiny_clm.model import export_model, load_export
from tiny_clm.provenance import INFERENCE_RUNTIME, verify_snapshot

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 8])
def test_teacher_forced_cache_positions_and_export_comparison(tmp_path, dtype, steps):
    torch.manual_seed(3)
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, tie_word_embeddings=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype, attn_implementation="sdpa").eval()
    ids = torch.arange(1, 25).reshape(1, -1)
    with sdpa_kernel(SDPBackend.MATH):
        native = capture(model, ids, steps)
    export_model(model, tmp_path/"export", {}, dtype=str(dtype).split(".")[-1])
    exported, _ = load_export(tmp_path/"export")
    with sdpa_kernel(SDPBackend.MATH):
        after = capture(exported, ids, steps)
    for path in native:
        assert native[path].shape == (steps, 128)
        assert comparison(after[path], native[path], rtol=0, atol=0)["allclose"]
    if dtype == torch.float32:
        assert comparison(native["explicit"], native["full"])["allclose"]
    assert comparison(native["implicit"], native["explicit"])["allclose"]


def test_cross_path_tolerance_cannot_mask_export_changes():
    reference = torch.tensor([[1.0, 2.0, 3.0]])
    changed = reference+0.01
    assert comparison(changed, reference, rtol=.02, atol=.25)["allclose"]
    assert not comparison(changed, reference)["allclose"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_comparison_cannot_pass(bad):
    result = comparison(torch.tensor([[bad, 0.0]]), torch.zeros(1, 2))
    assert not result["finite"] and not result["allclose"]


def fixture_source(tmp_path):
    old = tmp_path/"old"
    (old/"export").mkdir(parents=True)
    (old/"export/model.safetensors").write_bytes(b"fixture, not real weights")
    sha = file_sha(old/"export/model.safetensors")
    config = {"model": {"id": "fixture", "revision": "a"*40}}
    source = {"stage": "backbone_screen", "campaign": str(old), "config": config,
              "venv": "/fixture/env", "environment": {"freeze_sha256": "unchanged"}}
    audit = {"reload_passed": True, "cache_passed": False, "identical_parameters_and_buffers": True,
             "inference_runtime": INFERENCE_RUNTIME, "weight_sha256": sha, "model": config["model"]}
    artifact = {"inference_runtime": INFERENCE_RUNTIME, "weight_sha256": sha, "ranks": {}}
    for name, data in (("manifest.json", source), ("config.json", config), ("prepared.json", {}),
                       ("export-audit.json", audit), ("export/compression.json", artifact)):
        (old/name).write_text(json.dumps(data))
    return old


def test_failed_cache_gate_can_be_diagnosed_without_modifying_it(tmp_path):
    old = fixture_source(tmp_path)
    original = {str(p): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    _, audit = validate_source(old/"manifest.json", verify_weights=True)
    assert audit["cache_passed"] is False
    assert original == {str(p): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    (old/"export/model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_source(old/"manifest.json", verify_weights=True)


def test_diagnostic_dry_run_and_single_job_snapshot(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"compression"))
    module_spec = importlib.util.spec_from_file_location("cache_audit_driver", ROOT/"compression/cache_audit.py")
    driver = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(driver)
    repo = tmp_path/"repo"
    repo.mkdir()
    (repo/"code.py").write_text("# fixture\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    monkeypatch.setattr(driver, "ROOT", repo)
    monkeypatch.setattr(sys.modules["submit"], "ROOT", repo)
    old = fixture_source(tmp_path)
    original = {str(p): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    for key, value in {"TINY_CLM_WORK": str(tmp_path), "TINY_CLM_SNAPSHOTS": str(tmp_path/"snapshots"),
                       "TINY_CLM_VENV": "/fixture/env", "TINY_CLM_PARTITION": "gpu-fixture",
                       "TINY_CLM_GPU_CONSTRAINT": "fixture", "TINY_CLM_GPUS": "1",
                       "SLURM_JOB_ID": "parent", "SBATCH_GRES": "gpu:1"}.items():
        monkeypatch.setenv(key, value)
    verified, calls = [], []
    def verify(venv, expected):
        verified.append(venv)
        return expected
    monkeypatch.setattr(driver, "verify_environment", verify)
    real = subprocess.check_output
    def scheduler(command, **kw):
        if command[0] != "sbatch":
            return real(command, **kw)
        assert verified == ["/fixture/env"]
        assert "SLURM_JOB_ID" not in kw["env"] and "SBATCH_GRES" not in kw["env"]
        calls.append(command)
        return "123\n"
    monkeypatch.setattr(subprocess, "check_output", scheduler)
    destination = tmp_path/"diagnostic"
    args = ["--source-manifest", str(old/"manifest.json"), "--campaign", str(destination)]
    driver.main(args)
    assert not destination.exists() and not calls and not verified
    driver.main(args+["--submit"])
    spec = json.loads((destination/"manifest.json").read_text())
    verify_snapshot(spec["snapshot"], spec["snapshot_hash"])
    assert len(calls) == 1 and "--gpus=1" in calls[0] and "--mem=64G" in calls[0]
    assert calls[0][-1] == "cache_audit" and not any("--array=" in a for a in calls[0])
    assert original == {str(p): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        driver.main(args+["--submit"])
