"""Slurm stages for a bounded, uncompressed backbone screen (no optimizer)."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .io import exclusive, file_sha, memory, metadata, read_json, seed_all, write_json
from .provenance import verify_snapshot
from .screen_contract import qwen3_counts, screen_report, validate_screen


def verify_model_lock(path, config):
    lock = read_json(path)
    if lock["model"] != config["model"] or not lock["files"]:
        raise ValueError("model-only asset lock differs from screen request")
    source = Path(lock["source"])
    current = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    if current != set(lock["files"]):
        raise ValueError("model snapshot file inventory changed")
    for name, sha in lock["files"].items():
        if file_sha(source/name) != sha:
            raise ValueError("model snapshot file changed: "+name)
    return lock


def prepare(spec):
    from huggingface_hub import snapshot_download
    root = Path(spec["screen_assets"])
    root.mkdir(parents=True, exist_ok=True)
    path = root/"model-lock.json"
    with exclusive(root/".prepare.lock"):
        if path.exists():
            lock = verify_model_lock(path, spec["config"])
        else:
            hf_home = Path(os.environ["HF_HOME"])
            hf_home.mkdir(parents=True, exist_ok=True)
            # Download + export + bounded trajectories. This is available disk,
            # not a project quota check; users must also check their allocation.
            for location in (hf_home, Path(spec["campaign"])):
                if shutil.disk_usage(location).free < 40*2**30:
                    raise RuntimeError("need at least 40 GiB free for the diagnostic at "+str(location))
            model = spec["config"]["model"]
            source = Path(snapshot_download(repo_id=model["id"], revision=model["revision"],
                          allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
                          max_workers=min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))))
            if source.name != model["revision"]:
                raise ValueError("download did not resolve to the pinned model revision")
            files = {str(p.relative_to(source)): file_sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
            if not any(n.endswith(".safetensors") for n in files) or "config.json" not in files:
                raise ValueError("download lacks model weights/configuration")
            config = read_json(source/"config.json")
            for key, expected in spec["config"]["screen"]["audited_model_config"].items():
                if config.get(key) != expected:
                    raise ValueError(f"downloaded architecture changed at {key}")
            lock = {"model": model, "source": str(source), "files": files,
                    "analytical_counts": qwen3_counts(config), "metadata": metadata()}
            index = source/"model.safetensors.index.json"
            if index.exists():
                lock["publisher_index_metadata"] = read_json(index).get("metadata")
            write_json(path, lock)
        write_json(Path(spec["campaign"])/"prepared.json", {
            "model_lock": str(path), "model_lock_sha256": file_sha(path),
            "model": lock["model"], "metadata": metadata()})
    print("SCREEN_MODEL_PREPARED", flush=True)


def model_lock(spec):
    prepared = read_json(Path(spec["campaign"])/"prepared.json")
    path = Path(prepared["model_lock"])
    if file_sha(path) != prepared["model_lock_sha256"]:
        raise ValueError("model lock changed after preparation")
    return verify_model_lock(path, spec["config"])


def export(spec):
    import torch
    from .model import dense_model, export_model, load_export, tokenizer_from
    from .lowrank import accounting

    if (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported() or
            torch.cuda.get_device_properties(0).total_memory < 32*2**30):
        raise RuntimeError("screen requires one allocated BF16 GPU with at least 32 GiB VRAM")
    campaign = Path(spec["campaign"])
    if (campaign/"export").exists():
        raise FileExistsError("preserve the existing artifact/audit and use a new screen campaign")
    lock = model_lock(spec)
    seed_all(0)
    start = time.perf_counter()
    model = dense_model(lock["source"], device="cuda", dtype="bfloat16").eval()
    counts = qwen3_counts(model.config.to_dict())
    if accounting(model, {})["original_parameters"] != counts["parameters"]:
        raise ValueError("loaded parameter count differs from audited Qwen3 formula")
    tokenizer = tokenizer_from(lock["source"])
    tool = {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Print READY using bash."}],
              tools=[tool], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "<tool_call>" not in prompt or '"name": "bash"' not in prompt:
        raise ValueError("native tokenizer did not render the declared tool schema")
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        before = model(input_ids=ids, use_cache=False).logits[:, -1].float().cpu()
    info = export_model(model, campaign/"export", {}, tokenizer=tokenizer, dtype="bfloat16",
               provenance={"model": spec["config"]["model"], "screen_only": True,
                           "commit": spec["commit"], "model_lock_sha256": file_sha(
                               Path(spec["screen_assets"])/"model-lock.json")})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_export(campaign/"export", device="cuda")
    with torch.inference_mode():
        after = reloaded(input_ids=ids, use_cache=False).logits[:, -1].float().cpu()
    torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-4)
    # Cache compatibility test uses fixed tokens; it is not a quality measurement.
    with torch.inference_mode():
        prefix = reloaded(input_ids=ids[:, :-1], use_cache=True)
        cached = reloaded(input_ids=ids[:, -1:], past_key_values=prefix.past_key_values,
                          use_cache=True).logits[:, -1].float().cpu()
    # BF16 SDPA may accumulate differently for a one-token cached query.
    torch.testing.assert_close(cached, after, rtol=0.02, atol=0.25)
    write_json(campaign/"export-audit.json", {
        "model": spec["config"]["model"], "weight_sha256": info["weight_sha256"],
        "weight_bytes": info["weight_bytes"], "reload_passed": True,
        "reload_max_abs_error": (after-before).abs().max().item(),
        "reload_tolerance": {"rtol": 1e-4, "atol": 1e-4},
        "cache_max_abs_error": (cached-after).abs().max().item(),
        "cache_tolerance": {"rtol": 0.02, "atol": 0.25},
        "template_tools_present": True, "counts": counts, "whole_keep": 1.0,
        "elapsed_seconds": time.perf_counter()-start, "memory": memory(), "metadata": metadata(),
        "scope": "uncompressed BF16 loading/export/cache checks; no language or CLM quality claim"})
    print("SCREEN_EXPORT_VERIFIED", flush=True)


def main():
    from .environment import verify_environment
    manifest, stage = Path(sys.argv[1]), sys.argv[2]
    spec = read_json(manifest)
    campaign, snapshot = Path(spec["campaign"]), Path(spec["snapshot"])
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run screen stages inside a Slurm allocation")
    verify_snapshot(snapshot, spec["snapshot_hash"])
    verify_environment(spec["venv"], spec["environment"])
    if read_json(campaign/"config.json") != spec["config"] or spec["stage"] != "backbone_screen":
        raise ValueError("screen manifest/config mismatch")
    for key in ("reference_manifest", "contextbench_lock"):
        if file_sha(spec[key]) != spec[key+"_sha256"]:
            raise ValueError(key+" changed after submission")
    lock = read_json(spec["contextbench_lock"])
    validate_screen(spec["config"], read_json(spec["reference_manifest"]), lock)
    if stage == "screen_check":
        from .contextbench import verify_lock
        verify_lock(lock, spec["config"]["contextbench"])
        commands = [[sys.executable, "-m", "pytest", "-q", "compression/tests", "clm/tests/test_context_string.py",
                     "--junitxml="+str(campaign/"correctness.xml"), "-o", "cache_dir="+str(campaign/"pytest-cache")],
                    [sys.executable, str(snapshot/"compression/check_client_protocol.py")],
                    [sys.executable, "-m", "tiny_clm.smoke", "--output", str(campaign/"smoke")]]
        for cmd in commands:
            subprocess.run(cmd, check=True)
        write_json(campaign/"checks.json", {"passed": True, "metadata": metadata()})
    elif stage == "screen_prepare":
        if not read_json(campaign/"checks.json")["passed"]:
            raise RuntimeError("screen correctness checks did not pass")
        prepare(spec)
    elif stage == "screen_export":
        export(spec)
    elif stage == "screen_contextbench":
        from .contextbench import run
        audit = read_json(campaign/"export-audit.json")
        if not audit["reload_passed"] or file_sha(campaign/"export/model.safetensors") != audit["weight_sha256"]:
            raise ValueError("export is not verified")
        run(campaign/"export", spec["config"], spec["contextbench_lock"], campaign/"contextbench",
            task_names=spec["config"]["screen"]["task_names"],
            server_timeout=spec["config"]["screen"]["server_start_timeout"])
    elif stage == "screen_report":
        def optional(path):
            return read_json(path) if path.exists() else None
        report = screen_report(spec["config"], lock, optional(campaign/"export-audit.json"),
                               optional(campaign/"contextbench/summary.json"))
        report.update(commit=spec["commit"], manifest_sha256=file_sha(manifest), metadata=metadata())
        write_json(campaign/"screen-report.json", report)
        print(json.dumps(report, indent=2))
        if not report["complete"]:
            raise RuntimeError("incomplete screen; see screen-report.json and stage logs")
    else:
        raise ValueError("unknown screen stage: "+stage)


if __name__ == "__main__":
    main()
