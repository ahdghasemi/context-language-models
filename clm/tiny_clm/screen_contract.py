"""Dependency-free contracts for an inference-only backbone diagnostic."""
from __future__ import annotations

import math
from pathlib import Path

from .provenance import canonical_hash, select_task_records, INFERENCE_RUNTIME


def validate_screen(config, reference, lock):
    if set(config) != {"model", "export_dtype", "contextbench", "screen"}:
        raise ValueError("screen config must not contain training/data/rank settings")
    if config["export_dtype"] != "bfloat16" or config["screen"].get("no_training") is not True:
        raise ValueError("this diagnostic requires BF16 inference and no training")
    timeout = config["screen"]["server_start_timeout"]
    if type(timeout) is not int or not 1 <= timeout <= 1800:
        raise ValueError("server startup timeout must be 1..1800 seconds")
    if config["contextbench"] != reference["config"]["contextbench"]:
        raise ValueError("screen must retain the reference campaign's complete task/decoding contract")
    if config["contextbench"]["modes"] != ["clm", "append_only"]:
        raise ValueError("screen requires both original harness modes")
    revision = config["model"]["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("model revision must be an immutable lowercase commit SHA")
    floor = config["screen"]["min_clm_reward"]
    if not isinstance(floor, (int, float)) or not math.isfinite(floor) or not 0 < floor <= 1:
        raise ValueError("invalid provisional screening floor")
    if not lock["runnable"] or lock["request_hash"] != canonical_hash(config["contextbench"]):
        raise ValueError("reference task lock is not runnable or has a different protocol")
    return select_task_records(lock["tasks"], config["screen"]["task_names"])


def qwen3_counts(c):
    """Audited biasless Qwen3 architecture; count tied embeddings only once."""
    if c["model_type"] != "qwen3" or c["attention_bias"] or not c["tie_word_embeddings"]:
        raise ValueError("screen resource formula requires audited biasless, tied Qwen3")
    h, i, d, n = c["hidden_size"], c["intermediate_size"], c["head_dim"], c["num_hidden_layers"]
    q, k, v = c["num_attention_heads"], c["num_key_value_heads"], c["vocab_size"]
    mlp = n * 3 * h * i
    p = v * h + n * (3*h*i + 2*h*d*(q+k) + 2*h + 2*d) + h
    return {"parameters": p, "bf16_parameter_bytes": 2*p, "all_mlp_weights": mlp,
            "fp32_model_grad_adam_bytes": 16*p,
            "all_mlp_proxy_bytes": 4*mlp, "penalty_saved_difference_bytes": 4*mlp,
            "training_subtotal_excluding_workspaces_bytes": 16*p + 8*mlp}


def screen_report(config, lock, audit, summary):
    """No screen, even a passing one, is accepted as a baseline/pilot gate."""
    tasks = select_task_records(lock["tasks"], config["screen"]["task_names"])
    expected = {(m, Path(t["path"]).name): t["sha256"]
                for m in config["contextbench"]["modes"] for t in tasks}
    errors, records, seen = [], [], set()
    if (not audit or not audit.get("reload_passed") or not audit.get("cache_passed") or
            audit.get("inference_runtime") != INFERENCE_RUNTIME or audit.get("model") != config["model"]):
        errors.append("missing or incompatible export audit")
    if (not summary or not summary.get("all_complete") or
            summary.get("contextbench") != config["contextbench"] or
            summary.get("screen_task_names") != config["screen"]["task_names"] or
            summary.get("inference_runtime") != INFERENCE_RUNTIME or
            summary.get("container_sha256") != lock["image_sha256"] or
            summary.get("source_revision") != lock["revision"]):
        errors.append("missing, incomplete, or incompatible ContextBench summary")
    for ep in (summary or {}).get("episodes", []):
        key = (ep.get("mode"), ep.get("episode"))
        rewards = ep.get("rewards", {})
        reward = rewards.get("reward")
        valid = isinstance(reward, (int, float)) and math.isfinite(reward) and 0 <= reward <= 1
        if (key not in expected or key in seen or ep.get("status") != "complete" or ep.get("error") or
                ep.get("returncode") != 0 or rewards.get("grader_valid") != 1 or not valid or
                ep.get("weight_sha256") != (audit or {}).get("weight_sha256") or
                ep.get("inference_runtime") != INFERENCE_RUNTIME or
                ep.get("task_sha256") != expected.get(key) or
                ep.get("request_hash") != canonical_hash(config["contextbench"])):
            errors.append("invalid episode: " + repr(key))
        seen.add(key)
        records.append({k: ep.get(k) for k in ("episode", "mode", "status", "rewards", "error",
            "model_calls", "tool_calls", "prompt_tokens", "completion_tokens", "elapsed_seconds")})
    if seen != set(expected):
        errors.append("episode coverage differs from the declared screen subset")
    complete = not errors
    clm = [e["rewards"]["reward"] for e in records if e["mode"] == "clm"] if complete else []
    return {"gate_type": "backbone_screen", "inference_runtime": INFERENCE_RUNTIME,
            "passed": complete and bool(clm) and
                min(clm) >= config["screen"]["min_clm_reward"],
            "complete": complete, "errors": errors, "episodes": records,
            "model": config["model"], "can_advance_to_compression": False,
            "threshold_provisional": True, "min_clm_reward": config["screen"]["min_clm_reward"],
            "scope": "one diagnostic subset; a pass permits a full baseline and memory audit only; "
                     "no compression, language-quality, speedup, or population-level claim"}
