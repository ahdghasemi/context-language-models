from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_model, save_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .io import read_json, write_json, file_sha
from .lowrank import replace, accounting

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


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
    dst = Path(destination)
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite deployment artifact: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".export-", dir=dst.parent))
    try:
        model.to(device="cpu", dtype=DTYPES[dtype]).eval()
        model.config.save_pretrained(temp)
        model.generation_config.save_pretrained(temp)
        if tokenizer is not None:
            tokenizer.save_pretrained(temp)
        save_model(model, str(temp/"model.safetensors"))
        report = accounting(model, ranks)
        report.update(format="tiny-clm-factorized-v1", ranks=ranks, dtype=dtype,
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
    if spec["format"] != "tiny-clm-factorized-v1":
        raise ValueError("unsupported artifact format")
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
    return model.to(device=device, dtype=DTYPES[dtype or spec["dtype"]]).eval(), spec


def tokenizer_from(path, revision=None):
    return AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=False)
