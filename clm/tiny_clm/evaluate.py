from __future__ import annotations

import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig

from .data import load_blocks
from .io import digest, memory, metadata, read_json, seed_all, write_json
from .lowrank import replace
from .model import deployment_precision, export_model, load_export, tokenizer_from
from .provenance import INFERENCE_RUNTIME, LEGACY_INFERENCE_RUNTIME
from .train import nll


def load_run(run, device="cpu"):
    root = Path(run)
    state = torch.load(root/"latest.pt", map_location="cpu", weights_only=False)
    if not state["complete"] or read_json(root/"status.json")["status"] != "complete":
        raise RuntimeError("training is incomplete; export refused")
    cfg = state["config"]
    model_cfg = AutoConfig.from_pretrained(cfg["assets"]["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_config(model_cfg, attn_implementation="sdpa", torch_dtype=torch.float32)
    model.generation_config = GenerationConfig.from_pretrained(cfg["assets"]["model_path"], local_files_only=True)
    if state["factorized"]:
        replace(model, state["ranks"], initialize=False)
    model.load_state_dict(state["model"], strict=True)
    ranks, is_factorized = state["ranks"], state["factorized"]
    del state
    model.config.use_cache = True
    return model.to(device).eval(), cfg, ranks, is_factorized


@torch.no_grad()
def export_run(run, device="cpu"):
    root = Path(run)
    model, cfg, ranks, already_factorized = load_run(root, device)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    val = load_blocks(cfg["assets"]["files"]["validation"]["path"], model.config.vocab_size)
    dtype = cfg["export_dtype"]
    # Project the actual trained FP32 weights, as in the post-hoc controls.
    # Preserve only selected matrices while measuring the dense model at export precision.
    originals = ({name: model.get_submodule(name).weight.detach().cpu().clone() for name in ranks}
                 if ranks and not already_factorized else {})
    # Hold precision fixed on both sides of export; report rounding separately from training precision.
    deployment_precision(model, dtype)
    before = nll(model, val, cfg["training"]["validation_batches"])
    x = torch.tensor(val[:1].copy(), device=device, dtype=torch.long)
    old_logits = model(x, use_cache=False).logits.float().cpu()
    if ranks and not already_factorized:
        for name, weight in originals.items():
            layer = model.get_submodule(name)
            layer.weight = torch.nn.Parameter(weight.to(layer.weight.device), requires_grad=False)
        replace(model, ranks)
        deployment_precision(model, dtype)
    del originals
    after = nll(model, val, cfg["training"]["validation_batches"])
    new_logits = model(x, use_cache=False).logits.float().cpu()
    gap = {"before": before, "after": after, "nll_delta": after["nll"]-before["nll"],
           "inference_runtime": INFERENCE_RUNTIME,
           "perplexity_ratio": math.exp(after["nll"]-before["nll"]), "dtype": dtype,
           "logit_rmse_first_block": (old_logits-new_logits).square().mean().sqrt().item(),
           "argmax_agreement_first_block": (old_logits.argmax(-1)==new_logits.argmax(-1)).float().mean().item(),
           "note": "validation deployment gap, not evidence of CLM capability"}
    tok = tokenizer_from(cfg["assets"]["model_path"])
    if (root/"export").exists():
        report = read_json(root/"export/compression.json")
        if report.get("inference_runtime") != INFERENCE_RUNTIME:
            raise ValueError("existing export uses legacy precision; preserve it and use a new run")
        if digest(report["provenance"]["config"]) != digest(cfg) or report["ranks"] != ranks:
            raise ValueError("existing export belongs to a different configuration")
    else:
        report = export_model(model, root/"export", ranks, tokenizer=tok, dtype=dtype,
            provenance={"config": cfg, "training": read_json(root/"status.json"),
                        "metadata": read_json(root/"metadata.json"), "deployment_gap": gap})
    # A successful save must also reload and preserve logits.
    del model
    reloaded, _ = load_export(root/"export", device)
    got = reloaded(x, use_cache=False).logits.float().cpu()
    error = (got-new_logits).abs().max().item()
    if not torch.equal(got, new_logits):
        torch.testing.assert_close(got, new_logits, rtol=1e-4, atol=1e-4)
    gap["save_reload_max_logit_error"] = error
    write_json(root/"deployment_gap.json", gap)
    return report


def evaluate(artifact, output, split="validation", final_selection=None, device="cpu"):
    model, spec = load_export(artifact, device)
    cfg = spec["provenance"]["config"]
    if split == "test":
        if final_selection is None:
            raise ValueError("test evaluation requires --selection-record frozen before test access")
        selection = read_json(final_selection)
        if spec["weight_sha256"] not in selection["selected_weight_sha256"]:
            raise ValueError("artifact not in frozen selection record")
    from .assets import verify
    verify(cfg["assets"], splits=(split,))
    data = load_blocks(cfg["assets"]["files"][split]["path"], model.config.vocab_size)
    start = time.perf_counter()
    result = nll(model, data)
    result.update(split=split, artifact_sha256=spec["weight_sha256"], elapsed_seconds=time.perf_counter()-start,
                  inference_runtime=spec.get("inference_runtime", LEGACY_INFERENCE_RUNTIME),
                  metadata=metadata(), memory=memory(), scope="general LM teacher-forced NLL only")
    write_json(output, result)
    return result


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def summarize(values):
    return {"median": statistics.median(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values)>1 else 0., "samples": values}


@torch.no_grad()
def profile(artifact, output, device="cpu", lengths=(128, 512), decode_tokens=32,
            repeats=5, warmup=2, batch_size=1):
    if repeats < 2 or warmup < 1 or decode_tokens < 1 or batch_size < 1:
        raise ValueError("profile requires repeated measurements, warmup and positive sizes")
    model, spec = load_export(artifact, device)
    seed_all(123)
    model.config.use_cache = True
    cfg = spec["provenance"]["config"]
    data = load_blocks(cfg["assets"]["files"]["validation"]["path"], model.config.vocab_size).reshape(-1)
    workloads = []
    for length in lengths:
        if length+decode_tokens > model.config.max_position_embeddings or length*batch_size > len(data):
            raise ValueError("profile length exceeds available real tokens or model limit")
        x = torch.tensor(np.asarray(data[:length*batch_size]).copy(), device=device,
                         dtype=torch.long).reshape(batch_size, length)
        prefill, decode, cache_bytes = [], [], 0
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for i in range(warmup+repeats):
            sync(device)
            start = time.perf_counter()
            out = model(x, use_cache=True)
            sync(device)
            p = time.perf_counter()-start
            past = out.past_key_values
            token = out.logits[:, -1:].argmax(-1)
            del out
            sync(device)
            start = time.perf_counter()
            for _ in range(decode_tokens):
                out = model(token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                token = out.logits[:, -1:].argmax(-1)
            sync(device)
            d = time.perf_counter()-start
            # Native Qwen3 DynamicCache; explicitly measure persistent K/V tensors.
            legacy = past.to_legacy_cache()
            cache_bytes = sum(t.numel()*t.element_size() for pair in legacy for t in pair)
            del past, legacy, out
            if i >= warmup:
                prefill.append(p)
                decode.append(d)
        workloads.append({"prompt_tokens": length, "batch_size": batch_size,
            "decode_steps": decode_tokens, "prefill_seconds": summarize(prefill),
            "decode_seconds": summarize(decode),
            "prefill_tokens_per_second": batch_size*length/statistics.median(prefill),
            "decode_tokens_per_second": batch_size*decode_tokens/statistics.median(decode),
            "kv_cache_bytes": cache_bytes, "memory": memory()})
    result = {"metadata": metadata(), "artifact_sha256": spec["weight_sha256"],
        "inference_runtime": spec.get("inference_runtime", LEGACY_INFERENCE_RUNTIME),
        "parameters": sum(p.numel() for p in model.parameters()),
        "resident_parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
        "serialized_weight_bytes": spec["weight_bytes"], "dtype": spec["dtype"],
        "backend": "transformers SDPA attention; sequential PyTorch factor linears",
        "warmup": warmup, "repeats": repeats, "workloads": workloads,
        "scope": "fixed-length autoregressive kernel workloads, forced length, not task latency",
        "activation_memory_note": "CUDA peak includes activations/workspaces/cache; not isolated activation memory"}
    write_json(output, result)
    return result
