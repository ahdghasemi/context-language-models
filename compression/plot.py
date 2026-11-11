#!/usr/bin/env python3
"""Plot measured validation tradeoffs; missing measurements stay missing."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tiny_clm.io import read_json

p = argparse.ArgumentParser()
p.add_argument("--summary", required=True)
a = p.parse_args()
root = Path(a.summary)
rows = [r for r in read_json(root/"runs.json") if r["status"] == "complete"]
if not rows:
    raise SystemExit("No complete measured runs to plot")
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for method in sorted({r["method"] for r in rows}):
    data = [r for r in rows if r["method"] == method]
    for ax, key, label in zip(axes, ["perplexity", "clm_reward", "len512_decode_tps"],
                              ["Validation perplexity", "CLM reward (macro episodes)", "Decode tokens/s, 512-token prompt"]):
        valid = [r for r in data if r.get(key) is not None]
        ax.scatter([r["whole_keep"] for r in valid], [r[key] for r in valid], label=method)
        ax.set(xlabel="Whole-model parameter fraction", ylabel=label)
        ax.grid(alpha=.2)
axes[0].legend(fontsize=7)
fig.savefig(root/"tradeoffs.png", dpi=180)
fig.savefig(root/"tradeoffs.pdf")
