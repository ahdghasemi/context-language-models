#!/usr/bin/env python3
"""Read-only real-backbone audit; this is not compression training."""
import argparse
import os
import time

import torch

from tiny_clm.assets import verify
from tiny_clm.data import load_blocks
from tiny_clm.io import memory, metadata, read_json, write_json
from tiny_clm.lowrank import accounting, rank_for, select
from tiny_clm.model import dense_model
from tiny_clm.train import nll

p = argparse.ArgumentParser()
p.add_argument("--assets", required=True)
p.add_argument("--output", required=True)
p.add_argument("--device", default="cpu")
p.add_argument("--svd", action="store_true", help="exact full reduced SVD audit of all MLP matrices")
a = p.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
assets = read_json(a.assets)
verify(assets, model_weights=True)
start = time.perf_counter()
model = dense_model(assets["model_path"], device=a.device)
val = load_blocks(assets["files"]["validation"]["path"], model.config.vocab_size)
result = {"scope": "uncompressed real-model CPU/GPU audit; first two validation blocks only; no training",
          "model_revision": assets["model_revision"], "dataset_revision": assets["dataset_revision"],
          "nll": nll(model, val, batches=2), "metadata": metadata(), "budgets": {}}
for keep in [.8, .6, .4]:
    report = accounting(model, select(model, {"kind": "mlp", "keep": keep}))
    report.pop("layers")
    result["budgets"][str(keep)] = report
result["load_and_nll_seconds"] = time.perf_counter()-start
if a.svd:
    start = time.perf_counter()
    tails = {str(k): 0. for k in [.8, .6, .4]}
    for name in select(model, {"kind": "mlp", "keep": .8}):
        w = model.get_submodule(name).weight.detach().float()
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        for keep in [.8, .6, .4]:
            tails[str(keep)] += s[rank_for(w.shape, keep):].square().sum().item()
        del u, s, vh
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    result["svd"] = {"seconds": time.perf_counter()-start, "tail_squared_sum": tails,
                     "penalty_by_mu": {k: {str(mu): v/(2*mu) for mu in [10000, 1000, 100]} for k, v in tails.items()}}
result["memory"] = memory()
write_json(a.output, result)
print("Wrote read-only audit:", a.output)
