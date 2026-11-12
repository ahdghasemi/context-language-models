"""Short resource probe using the real trainer; random tokens, no quality claim."""
from __future__ import annotations

import copy
import shutil
import tempfile
import time
from pathlib import Path

from .screen_contract import qwen3_counts


def probe_config(screen):
    return {"model": copy.deepcopy(screen["model"]), "seed": 0, "method": "constrained",
        "selection": {"kind": "down", "keep": 0.6},
        "penalty": {"initial": 10000.0, "minimum": 100.0, "gamma": 0.1},
        "training": {"stage_steps": 2, "batch_size": 1, "accumulation": 4,
            "eval_every": 2, "checkpoint_every": 2, "validation_batches": 1,
            "lr": 1e-5, "precision": "bfloat16", "gradient_checkpointing": True,
            "projection_every": 1, "optimizer_policy": "gnn_persistent", "adam_foreach": False}}


def estimates(screen):
    c = screen["screen"]["audited_model_config"]
    counts = qwen3_counts(c)
    selected = c["num_hidden_layers"] * c["hidden_size"] * c["intermediate_size"]
    rank = int(0.6*c["hidden_size"]*c["intermediate_size"]/(c["hidden_size"]+c["intermediate_size"]))
    factors = c["num_hidden_layers"] * rank * (c["hidden_size"]+c["intermediate_size"])
    return {"scope": "analytical bytes; excludes activations, SVD/Adam workspaces and allocator overhead",
        "parameters": counts["parameters"], "all_mlp_subtotal_bytes": counts["training_subtotal_excluding_workspaces_bytes"],
        "down_subtotal_bytes": 16*counts["parameters"]+8*selected,
        "down_selected_weights": selected, "down_rank": rank,
        "predicted_deployment_parameters": counts["parameters"]-selected+factors,
        "predicted_whole_reduction": (selected-factors)/counts["parameters"],
        "all_parameters_trainable": True, "minimum_measured_headroom_bytes": 4*2**30,
        "temporary_disk_required_bytes": 160*2**30}


def probe(artifact, screen, output):
    """Exercise two updates, a second projection with Adam resident, validation,
    best-weight CPU storage, and atomic checkpoint I/O. Delete temporary weights.
    This deliberately does not claim full-run, resume, or larger-workload safety.
    """
    import numpy as np
    import torch
    from .io import memory, metadata, read_json, write_json
    from .lowrank import accounting, select
    from .model import load_export
    from .train import train

    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve previous memory measurements")
    cfg, estimate = probe_config(screen), estimates(screen)
    report = {"probe_complete": False, "passed": False, "error": None,
        "scope": "two full-trainer updates on random tokens using pretrained weights; memory only, no quality result",
        "training_config": cfg, "sequence_length": 512, "estimates": estimate,
        "model": screen["model"], "metadata": metadata(), "training_artifact_retained": False,
        "can_advance_to_compression": False}
    start = time.perf_counter()
    write_json(output, report)
    try:
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("requires exactly one allocated BF16 CUDA GPU")
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        report.update(device_total_bytes=total, device_free_before_bytes=free)
        if free < estimate["down_subtotal_bytes"] + 8*2**30:
            raise RuntimeError("insufficient free VRAM for down-only probe plus provisional 8 GiB workspace allowance")
        if shutil.disk_usage(tempfile.gettempdir()).free < estimate["temporary_disk_required_bytes"]:
            raise RuntimeError("probe needs 160 GiB free in TMPDIR for atomic training checkpoints")
        model, exported = load_export(artifact, device="cuda", dtype="float32")
        report["weight_sha256"] = exported["weight_sha256"]
        if (sum(p.numel() for p in model.parameters()) != estimate["parameters"] or
                not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())):
            raise ValueError("probe must keep every audited model parameter trainable in FP32")
        report["budget"] = accounting(model, select(model, cfg["selection"]))
        rng = np.random.default_rng(cfg["seed"])
        tokens = rng.integers(0, model.config.vocab_size, size=(9, 512), dtype=np.int64)
        with tempfile.TemporaryDirectory(prefix="training-memory-") as folder:
            try:
                finished = train(model, tokens[:8], tokens[8:], cfg, folder, stop_after=2)
                report["trainer_status"] = read_json(Path(folder)/"status.json")
                report["checkpoint_bytes"] = (Path(folder)/"latest.pt").stat().st_size
                report["probe_complete"] = not finished and report["trainer_status"]["total_steps"] == 2
            finally:
                failure = Path(folder)/"failure.json"
                if failure.exists():
                    report["trainer_failure"] = read_json(failure)
        torch.cuda.synchronize()
        report["memory"] = memory()
        report["reserved_headroom_bytes"] = total-report["memory"]["cuda_peak_reserved"]
        report["passed"] = (report["probe_complete"] and
            report["reserved_headroom_bytes"] >= estimate["minimum_measured_headroom_bytes"])
        if not report["passed"]:
            report["error"] = "probe interrupted or less than provisional 4 GiB measured allocator headroom"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["elapsed_seconds"] = time.perf_counter()-start
        report["memory"] = memory()
        write_json(output, report)
    return report
