"""Random-weight/data software exercise. Never a compression-quality result."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from .evaluate import evaluate, export_run, profile
from .io import file_sha, seed_all, write_json
from .train import train


def tiny_model():
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        tie_word_embeddings=True, eos_token_id=2, bos_token_id=1, pad_token_id=0,
        max_position_embeddings=128, attention_dropout=0.0, _attn_implementation="sdpa"))


def tiny_config():
    return {"seed": 7, "method": "constrained", "selection": {"kind": "mlp", "keep": 0.5},
        "penalty": {"initial": 2.0, "minimum": 1.0, "gamma": 0.5}, "export_dtype": "float32",
        "training": {"stage_steps": 3, "batch_size": 1, "accumulation": 2,
                     "eval_every": 1, "checkpoint_every": 1, "validation_batches": 2,
                     "lr": 0.001, "precision": "float32", "gradient_checkpointing": False},
        "software_only": True}


def prepare_tiny(root):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = root/"source"
    seed_all(2)
    model = tiny_model()
    model.save_pretrained(source)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
             **{f"w{i}": i for i in range(4, 64)}}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", bos_token="[BOS]",
                         eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] or '' }} {% endfor %}"
    tokenizer.save_pretrained(source)
    rng = np.random.default_rng(6)
    files, arrays = {}, {}
    for split in ("train", "validation", "test"):
        arrays[split] = rng.integers(4, 64, (8, 16), dtype=np.int32)
        path = root/f"{split}.npy"
        np.save(path, arrays[split], allow_pickle=False)
        files[split] = {"path": str(path.resolve()), "sha256": file_sha(path)}
    cfg = tiny_config()
    cfg["assets"] = {"model_path": str(source.resolve()), "model_files": {}, "files": files,
                     "model_id": "random-tiny-qwen3", "model_revision": "synthetic",
                     "dataset_id": "random-token-software-fixture"}
    return model, cfg, arrays


def run(destination, device="cpu"):
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("smoke destination must be empty")
    torch.set_num_threads(2)
    model, cfg, data = prepare_tiny(root)
    model.to(device)
    result = train(model, data["train"], data["validation"], cfg, root/"run")
    assert result
    del model
    report = export_run(root/"run", device)
    quality = evaluate(root/"run/export", root/"run/evaluation.json", device=device)
    profile(root/"run/export", root/"run/profile.json", device, lengths=(8, 16), decode_tokens=3, repeats=3)
    write_json(root/"SMOKE_PASSED.json", {"passed": True, "software_only": True,
        "parameters": report["deployment_parameters"], "original_parameters": report["original_parameters"],
        "validation_nll_random_data": quality["nll"], "device": device,
        "scope": "random weights/tokens; no language or CLM capability claim"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    run(a.output, a.device)
