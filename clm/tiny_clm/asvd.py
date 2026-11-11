"""Separate activation-aware baseline with sensitivity-based rank allocation.

Mechanism follows hahnyuan/ASVD4LLM at 863042743edb4c0dcc512a45534dea07df16b140.
This implementation uses exact SVD, audited targets, actual parameter budgets,
and fails on decomposition errors. It does not implement ASVD KV-cache changes.
"""
from __future__ import annotations

import torch

from .lowrank import FactorizedLinear, rank_for
from .train import nll


def allocate(shapes, scores, target_keep):
    """Lowest isolated-NLL candidates first, monotonically decreasing ranks.

    Stops at the first attainable allocation at or below the exact target.
    This is the suffix-threshold policy in ASVD's binary search, with exact counts.
    """
    current = {name: None for name in shapes}
    dense = sum(m*n for m, n in shapes.values())
    count = dense
    for score, name, rank in sorted(scores):
        m, n = shapes[name]
        old = current[name]
        if old is not None and old <= rank:
            continue
        count -= (m*n if old is None else old*(m+n)) - rank*(m+n)
        current[name] = rank
        if count <= target_keep*dense:
            return {n: r for n, r in current.items() if r is not None}, count/dense
    raise ValueError("ASVD candidate grid cannot meet requested budget")


@torch.no_grad()
def sensitivity_ranks(model, targets, scales, calibration, batches, target_keep,
                      candidates=(.4, .5, .6, .7, .8, .9)):
    scores, shapes, records = [], {}, []
    model.eval()
    for name in targets:
        layer = model.get_submodule(name)
        m, n = layer.weight.shape
        shapes[name] = (m, n)
        parent, attr = name.rsplit(".", 1)
        parent = model.get_submodule(parent)
        scaled = layer.weight.detach().float()*scales[name]
        u, s, vh = torch.linalg.svd(scaled, full_matrices=False)
        try:
            for fraction in candidates:
                rank = rank_for((m, n), fraction)
                fac = FactorizedLinear(n, m, rank, layer.bias is not None,
                                       device=layer.weight.device, dtype=layer.weight.dtype)
                fac.left.weight.copy_(u[:, :rank]*s[:rank])
                fac.right.weight.copy_(vh[:rank]/scales[name])
                if layer.bias is not None:
                    fac.left.bias.copy_(layer.bias)
                setattr(parent, attr, fac)
                score = nll(model, calibration, batches)["nll"]
                scores.append((score, name, rank))
                records.append({"layer": name, "rank": rank, "keep": fraction, "calibration_nll": score})
        finally:
            setattr(parent, attr, layer)
        del scaled, u, s, vh
    ranks, actual = allocate(shapes, scores, target_keep)
    return ranks, {"actual_selected_keep": actual, "requested_selected_keep": target_keep,
                   "records": records, "calibration_batches": batches,
                   "scope": "ASVD sensitivity threshold allocation on training calibration data; exact SVD adaptation"}
