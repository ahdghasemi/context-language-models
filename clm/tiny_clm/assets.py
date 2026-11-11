"""One-time, revision-pinned acquisition; compute jobs never download assets."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .io import atomic_target, digest, exclusive, file_sha, read_json, write_json


def prepare(config, destination):
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    request = {"model": config["model"], "data": config["data"]}
    with exclusive(dest/".prepare.lock"):
        lock = dest/"assets-lock.json"
        if lock.exists():
            manifest = read_json(lock)
            if manifest["request_hash"] != digest(request):
                raise ValueError("asset directory belongs to a different model/data request")
            verify(manifest)
            return manifest
        api = HfApi()
        model_id = config["model"]["id"]
        model_rev = api.model_info(model_id, revision=config["model"].get("revision", "main")).sha
        model_path = snapshot_download(model_id, revision=model_rev,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer.model"])
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        d = config["data"]
        data_rev = api.dataset_info(d["id"], revision=d.get("revision", "main")).sha
        ds = load_dataset(d["id"], d["name"], revision=data_rev,
                          cache_dir=os.environ.get("HF_DATASETS_CACHE"))
        files = {}
        for split in ("train", "validation", "test"):
            max_tokens = d["max_tokens"][split]
            ids = []
            for row in ds[split]:
                text = row[d.get("text_column", "text")]
                if text.strip():
                    ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
                    ids.append(tokenizer.eos_token_id)
                if len(ids) >= max_tokens:
                    break
            block = d["sequence_length"]
            usable = min(len(ids), max_tokens)//block*block
            if usable < block:
                raise ValueError(f"not enough tokens in {split}")
            tokens = np.asarray(ids[:usable], dtype=np.int32).reshape(-1, block)
            path = dest/f"{split}.npy"
            with atomic_target(path) as tmp:
                with open(tmp, "wb") as f:
                    np.save(f, tokens, allow_pickle=False)
            files[split] = {"path": str(path), "sha256": file_sha(path),
                            "blocks": len(tokens), "tokens": usable}
        model_files = {str(p.relative_to(model_path)): file_sha(p) for p in Path(model_path).iterdir()
                       if p.is_file() and p.suffix in {".json", ".safetensors", ".jinja"}}
        manifest = {"format": 1, "request_hash": digest(request), "request": request,
                    "model_id": model_id, "model_revision": model_rev, "model_path": str(model_path),
                    "model_files": model_files, "dataset_id": d["id"], "dataset_config": d["name"],
                    "dataset_revision": data_rev, "sequence_length": d["sequence_length"],
                    "tokenizer_source": model_id, "files": files,
                    "split_policy": "upstream train/validation/test; concatenated documents separated by EOS"}
        write_json(lock, manifest)
        return manifest


def verify(manifest, splits=("train", "validation"), model_weights=False):
    # Test bytes are verified only when explicitly evaluating the final test split.
    for split in splits:
        record = manifest["files"][split]
        if file_sha(record["path"]) != record["sha256"]:
            raise ValueError(f"{split} tokens changed after preparation")
    for name, expected in manifest["model_files"].items():
        if model_weights or not name.endswith(".safetensors"):
            if file_sha(Path(manifest["model_path"])/name) != expected:
                raise ValueError(f"model asset changed: {name}")


def resolve(config, lock_path):
    manifest = read_json(lock_path)
    if digest({"model": config["model"], "data": config["data"]}) != manifest["request_hash"]:
        raise ValueError("configuration does not match prepared model/data")
    verify(manifest, model_weights=True)
    cfg = dict(config)
    cfg["assets"] = manifest
    return cfg, manifest
