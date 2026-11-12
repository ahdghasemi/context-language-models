from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import warnings

import torch
from safetensors.torch import load_model, save_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .io import read_json, write_json, file_sha
from .lowrank import replace, accounting, FactorizedLinear
from .provenance import INFERENCE_RUNTIME

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


@torch.no_grad()
def inference_state(model):
    """Hash unique parameter bytes and nonpersistent buffers, one tensor at a time."""
    parameters = hashlib.sha256()
    buffers = {}
    for kind, entries in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in entries:
            header = {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            if kind == "parameter":
                parameters.update(json.dumps(header, sort_keys=True).encode()+b"\0")
                parameters.update(memoryview(raw))
            else:
                buffers[name] = {**header, "sha256": hashlib.sha256(memoryview(raw)).hexdigest()}
    return {"parameter_sha256": parameters.hexdigest(), "buffers": buffers}


@torch.no_grad()
def deployment_precision(model, dtype, device=None):
    """Cast weights while preserving Qwen3's canonical FP32 RoPE frequencies.

    Module.to(dtype=...) casts nonpersistent buffers too. Upcasting an already
    rounded inv_freq does not recover its values; reject such inputs instead.
    Only the static/default RoPE used by the audited checkpoints is supported.
    """
    if model.config.model_type != "qwen3":
        raise ValueError("deployment precision is audited only for Qwen3")
    rotary = model.model.rotary_emb
    if rotary.rope_type != "default":
        raise ValueError("audit scaled/dynamic RoPE before exporting it")
    freq = rotary.inv_freq
    expected, _ = rotary.rope_init_fn(model.config, freq.device)
    if freq.dtype != torch.float32 or not torch.allclose(freq, expected, rtol=1e-6, atol=0):
        raise ValueError("RoPE frequencies were rounded or changed; reload the original model before export")
    # Keep the original tensor reference: Module.to replaces the buffer rather
    # than preserving its FP32 dtype. No large weight copy is retained here.
    model.to(device=device, dtype=DTYPES[dtype])
    rotary.inv_freq = freq.to(device=next(model.parameters()).device)
    rotary.original_inv_freq = rotary.inv_freq
    return model


def dense_model(source, revision=None, *, device="cpu", dtype="float32"):
    if not Path(source).is_dir() and (not revision or len(revision) != 40):
        raise ValueError("remote models require an immutable 40-character revision")
    config = AutoConfig.from_pretrained(source, revision=revision, trust_remote_code=False)
    if config.model_type != "qwen3":
        raise ValueError("only audited Qwen3ForCausalLM is supported")
    model = AutoModelForCausalLM.from_pretrained(source, revision=revision, config=config,
                torch_dtype=DTYPES[dtype], attn_implementation="sdpa", trust_remote_code=False)
    return model.to(device)


def export_model(model, destination, ranks, *, tokenizer=None, provenance=None, dtype="bfloat16"):
    """Atomic directory publication, safetensors, no optimizer/proxy/dense duplicates."""
    for name, rank in ranks.items():
        layer = model.get_submodule(name)
        if not isinstance(layer, FactorizedLinear) or layer.rank != rank:
            raise ValueError("export requires physical factors matching every declared rank")
    dst = Path(destination)
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite deployment artifact: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".export-", dir=dst.parent))
    try:
        deployment_precision(model, dtype, device="cpu").eval()
        model.config.save_pretrained(temp)
        model.generation_config.save_pretrained(temp)
        if tokenizer is not None:
            tokenizer.save_pretrained(temp)
        save_model(model, str(temp/"model.safetensors"))
        report = accounting(model, ranks)
        report.update(format="tiny-clm-factorized-v2", inference_runtime=INFERENCE_RUNTIME,
                      ranks=ranks, dtype=dtype,
                      provenance=provenance or {}, weight_bytes=(temp/"model.safetensors").stat().st_size,
                      weight_sha256=file_sha(temp/"model.safetensors"))
        write_json(temp/"compression.json", report)
        os.rename(temp, dst)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return report


def load_export(path, device="cpu", dtype=None):
    path = Path(path)
    spec = read_json(path/"compression.json")
    if spec["format"] not in {"tiny-clm-factorized-v1", "tiny-clm-factorized-v2"}:
        raise ValueError("unsupported artifact format")
    if spec["format"] == "tiny-clm-factorized-v2" and spec.get("inference_runtime") != INFERENCE_RUNTIME:
        raise ValueError("unsupported inference precision policy")
    if file_sha(path/"model.safetensors") != spec["weight_sha256"]:
        raise ValueError("deployment checkpoint checksum mismatch")
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    if cfg.model_type != "qwen3":
        raise ValueError("unsupported architecture")
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa",
                                           torch_dtype=DTYPES[spec["dtype"]])
    replace(model, spec["ranks"], initialize=False)
    load_model(model, str(path/"model.safetensors"), strict=True)
    model.generation_config = GenerationConfig.from_pretrained(path, local_files_only=True)
    if accounting(model, spec["ranks"])["deployment_parameters"] != spec["deployment_parameters"]:
        raise ValueError("parameter count does not match manifest")
    if spec["format"] == "tiny-clm-factorized-v1":
        warnings.warn("Legacy v1 artifact: reproducing whole-model dtype casts, including RoPE buffers; "
                      "re-export from source for corrected v2 comparisons", RuntimeWarning, stacklevel=2)
        model.to(device=device, dtype=DTYPES[dtype or spec["dtype"]])
    else:
        deployment_precision(model, dtype or spec["dtype"], device=device)
    return model.eval(), spec


def tokenizer_from(path, revision=None):
    return AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=False)
