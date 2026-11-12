"""Read-only native/export cache diagnostics; never authorizes a screen or training."""
from __future__ import annotations

from contextlib import nullcontext
import gc
from pathlib import Path
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .io import file_sha, memory, metadata, read_json, seed_all, write_json
from .model import dense_model, deployment_precision, inference_state, load_export, tokenizer_from
from .provenance import INFERENCE_RUNTIME
from .screen import model_lock


def comparison(actual, reference, *, rtol=1e-4, atol=1e-4):
    if actual.shape != reference.shape or actual.ndim != 2:
        raise ValueError("expected matching [positions, vocabulary] logits")
    actual, reference = actual.float(), reference.float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return {"finite": False, "allclose": False, "rtol": rtol, "atol": atol}
    delta = actual-reference
    log_p, log_q = reference.double().log_softmax(-1), actual.double().log_softmax(-1)
    p, q = log_p.exp(), log_q.exp()
    top = reference.topk(2, dim=-1).values
    return {"finite": True, "allclose": torch.allclose(actual, reference, rtol=rtol, atol=atol),
            "rtol": rtol, "atol": atol, "max_abs_error": delta.abs().max().item(),
            "rmse": delta.square().mean().sqrt().item(),
            "fraction_outside_tolerance": (delta.abs() > atol+rtol*reference.abs()).float().mean().item(),
            "argmax_agreement": (actual.argmax(-1) == reference.argmax(-1)).float().mean().item(),
            "reference_argmax_ids": reference.argmax(-1).tolist(),
            "actual_argmax_ids": actual.argmax(-1).tolist(),
            "reference_top1_margin_min": (top[:, 0]-top[:, 1]).min().item(),
            "max_kl_reference_to_actual": (p*(log_p-log_q)).sum(-1).max().item(),
            "max_total_variation": (0.5*(p-q).abs().sum(-1)).max().item()}


@torch.inference_mode()
def capture(model, ids, suffix_steps=1):
    """Teacher-force identical IDs through fresh implicit and explicit caches.

    Keep the full logits projection to reproduce the original screen's compute
    shapes. Only the suffix logits are retained on CPU; no KV state is returned.
    """
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= suffix_steps < ids.shape[1]:
        raise ValueError("expected one unpadded sequence longer than the suffix")
    length, prefix_length = ids.shape[1], ids.shape[1]-suffix_steps
    full = model(input_ids=ids, use_cache=False).logits[0, -suffix_steps:].float().cpu()
    result = {"full": full}
    for explicit in (False, True):
        prefix_kw = ({"attention_mask": torch.ones_like(ids[:, :prefix_length]),
                      "cache_position": torch.arange(prefix_length, device=ids.device)} if explicit else {})
        prefix = model(input_ids=ids[:, :prefix_length], use_cache=True, **prefix_kw)
        cache = prefix.past_key_values
        del prefix
        logits = []
        if int(cache.get_seq_length()) != prefix_length:
            raise ValueError("prefix cache length is incorrect")
        for pos in range(prefix_length, length):
            kw = ({"attention_mask": torch.ones_like(ids[:, :pos+1]),
                   "cache_position": torch.tensor([pos], device=ids.device)} if explicit else {})
            output = model(input_ids=ids[:, pos:pos+1], past_key_values=cache, use_cache=True, **kw)
            cache = output.past_key_values
            if int(cache.get_seq_length()) != pos+1:
                raise ValueError("cache did not advance by exactly one token")
            logits.append(output.logits[0, -1].float().cpu())
            del output
        result["explicit" if explicit else "implicit"] = torch.stack(logits)
        del cache
    return result


def within_model(values):
    return {"implicit_vs_full": comparison(values["implicit"], values["full"], rtol=.02, atol=.25),
            "explicit_vs_full": comparison(values["explicit"], values["full"], rtol=.02, atol=.25),
            "explicit_vs_implicit": comparison(values["explicit"], values["implicit"])}


def validate_source(manifest_path, *, verify_weights=False):
    """Accept an intact v2 uncompressed export, even if its cache gate failed."""
    source = read_json(manifest_path)
    root = Path(source["campaign"])
    if source["stage"] != "backbone_screen" or read_json(root/"config.json") != source["config"]:
        raise ValueError("source is not an intact backbone screen")
    audit = read_json(root/"export-audit.json")
    artifact = read_json(root/"export/compression.json")
    if (audit.get("inference_runtime") != INFERENCE_RUNTIME or not audit.get("reload_passed") or
            not audit.get("identical_parameters_and_buffers") or audit.get("model") != source["config"]["model"] or
            artifact.get("inference_runtime") != INFERENCE_RUNTIME or artifact.get("ranks") != {} or
            artifact.get("weight_sha256") != audit.get("weight_sha256")):
        raise ValueError("cache audit requires a verified, uncompressed v2 reload")
    if verify_weights and file_sha(root/"export/model.safetensors") != artifact["weight_sha256"]:
        raise ValueError("source export checksum changed")
    return source, audit


def run(spec):
    campaign = Path(spec["campaign"])
    output = campaign/"cache-audit.json"
    if output.exists():
        raise FileExistsError("preserve prior diagnostics and choose a new campaign")
    report = {"diagnostic_complete": False, "can_advance_to_compression": False,
              "authorizes_screen": False, "inference_runtime": INFERENCE_RUNTIME,
              "metadata": metadata(), "source_manifest_sha256": spec["source_manifest_sha256"],
              "scope": "cache implementation/numerical diagnostics only; no task quality or speed claim",
              "models": {}, "export_equivalence": {}}
    start = time.perf_counter()
    try:
        if (torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported() or
                torch.cuda.get_device_properties(0).total_memory < 32*2**30):
            raise RuntimeError("cache audit requires one BF16 GPU with at least 32 GiB")
        if file_sha(spec["source_manifest"]) != spec["source_manifest_sha256"]:
            raise ValueError("source manifest changed")
        for name, sha in spec["source_files"].items():
            if file_sha(name) != sha:
                raise ValueError("source evidence changed after submission: "+name)
        source, audit = validate_source(spec["source_manifest"], verify_weights=True)
        if source["config"] != spec["config"]:
            raise ValueError("source model/configuration changed")
        lock = model_lock(source)  # Verifies the pinned native checkpoint inventory/checksums.
        artifact = Path(source["campaign"])/"export"
        tokenizer = tokenizer_from(lock["source"])
        tool = {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                               "required": ["command"]}}}
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Print READY using bash."}],
                 tools=[tool], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        long_ids = ids.repeat(1, (2048+ids.shape[1]-1)//ids.shape[1])[:, :2048]
        cases = {"screen_last_token": (ids, 1), "screen_suffix_8": (ids, 8),
                 "synthetic_2048_suffix_8": (long_ids, 8)}
        report.update(model=source["config"]["model"], weight_sha256=audit["weight_sha256"],
                      input_ids=ids[0].tolist(), synthetic_long_input="screen prompt IDs repeated/truncated to 2048",
                      policy="default BF16 SDPA, forced-math BF16, then FP32 math with the same BF16-rounded weights")
        native = {}
        generation = {}
        seed_all(0)
        report["backend_settings"] = {"tf32": torch.backends.cuda.matmul.allow_tf32,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp": torch.backends.cuda.math_sdp_enabled()}
        for label in ("native", "exported"):
            model = (dense_model(lock["source"], device="cuda", dtype="bfloat16") if label == "native"
                     else load_export(artifact, device="cuda")[0]).eval()
            report["models"][label] = {"state": inference_state(model), "cases": {}}
            expected_state = audit["source_state"] if label == "native" else audit["reloaded_state"]
            if report["models"][label]["state"] != expected_state:
                raise ValueError("current model state differs from the verified export audit")
            for mode in ("bf16_default", "bf16_math", "fp32_math"):
                if mode == "fp32_math":
                    deployment_precision(model, "float32")
                selected = cases if mode == "bf16_default" else {"screen_last_token": cases["screen_last_token"]}
                for name, (tokens, steps) in selected.items():
                    key = mode+"/"+name
                    with nullcontext() if mode == "bf16_default" else sdpa_kernel(SDPBackend.MATH):
                        values = capture(model, tokens, steps)
                    report["models"][label]["cases"][key] = within_model(values)
                    if label == "native":
                        native[key] = values
                    else:
                        report["export_equivalence"][key] = {
                            path: comparison(values[path], native[key][path]) for path in values}
                    write_json(output, report)
                if mode == "bf16_default":
                    generation[label] = {}
                    for cached in (True, False):
                        with torch.inference_mode():
                            generated = model.generate(ids, attention_mask=torch.ones_like(ids),
                                max_new_tokens=16, do_sample=False, use_cache=cached,
                                pad_token_id=tokenizer.eos_token_id)
                        generation[label]["cached" if cached else "uncached"] = generated[0, ids.shape[1]:].tolist()
                    report["models"][label]["greedy_generation"] = generation[label]
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report["same_path_export_passed"] = all(
            m["allclose"] for paths in report["export_equivalence"].values() for m in paths.values())
        report["same_path_generation_equal"] = generation["native"] == generation["exported"]
        report["diagnostic_complete"] = True
        if not report["same_path_export_passed"] or not report["same_path_generation_equal"]:
            raise RuntimeError("native/export behavior differs; inspect cache-audit.json")
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-start, memory=memory())
        write_json(output, report)
    print("CACHE_AUDIT_COMPLETE: diagnostic only; existing screen gate remains unchanged", flush=True)


if __name__ == "__main__":
    run(read_json(sys.argv[1]))
