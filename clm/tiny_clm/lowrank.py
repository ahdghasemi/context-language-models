"""Hard-rank auxiliary projection and physical two-linear deployment.

This is the Constrained formulation, not the separately supplied nuclear-norm
algorithm. Penalties are SUMS of squared Frobenius norms, without normalization.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import torch
from torch import nn


def mu_schedule(initial: float, minimum: float, gamma: float) -> list[float]:
    if not all(math.isfinite(x) for x in (initial, minimum, gamma)):
        raise ValueError("non-finite penalty schedule")
    if not (initial >= minimum > 0 and 0 < gamma < 1):
        raise ValueError("require mu_init >= mu_min > 0 and 0 < gamma < 1")
    values = []
    mu = initial
    while mu >= minimum * (1 - 1e-12):
        values.append(mu)
        if len(values) > 1000:
            raise ValueError("more than 1000 penalty stages")
        mu *= gamma
    return values


def rank_for(shape, keep: float) -> int:
    m, n = shape
    if not math.isfinite(keep) or not 0 < keep < 1:
        raise ValueError("selected weight keep fraction must be in (0,1)")
    r = math.floor(keep * m * n / (m + n))
    validate_rank(m, n, r)
    return r


def validate_rank(m: int, n: int, r: int) -> None:
    if type(r) is not int or not 0 < r <= min(m, n) or r * (m + n) >= m * n:
        raise ValueError(f"rank {r} does not physically compress {m}x{n}")


@torch.no_grad()
def factors(weight: torch.Tensor, rank: int, scale=None):
    """Exact reduced SVD, hard truncation, and no gradient through the proxy.

    `scale` is ONLY for the separately named ASVD baseline.
    FP64 inputs retain FP64 for numerical tests; all other SVDs use FP32.
    """
    if weight.ndim != 2 or not 0 < rank <= min(weight.shape):
        raise ValueError("invalid projection rank/shape")
    w = weight.detach().to(torch.float64 if weight.dtype == torch.float64 else torch.float32)
    if not torch.isfinite(w).all():
        raise ValueError("non-finite matrix before SVD")
    if scale is not None:
        scale = scale.to(device=w.device, dtype=w.dtype)
        if scale.shape != (w.shape[1],) or not (scale > 0).all():
            raise ValueError("ASVD scale must be positive and match input channels")
        w = w * scale
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    a = (u[:, :rank] * s[:rank]).contiguous()
    b = vh[:rank].contiguous()
    if scale is not None:
        b = b / scale
    return a, b


@torch.no_grad()
def project(weight, rank):
    a, b = factors(weight, rank)
    return (a @ b).detach()


class FactorizedLinear(nn.Module):
    """y=A(Bx)+bias; no dense weight property or reconstruction in forward."""

    def __init__(self, in_features, out_features, rank, bias=True, *, device=None, dtype=None):
        super().__init__()
        validate_rank(out_features, in_features, rank)
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.right = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.left = nn.Linear(rank, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.left(self.right(x))

    @classmethod
    @torch.no_grad()
    def from_linear(cls, layer, rank, scale=None):
        a, b = factors(layer.weight, rank, scale)
        out = cls(layer.in_features, layer.out_features, rank, layer.bias is not None,
                  device=layer.weight.device, dtype=layer.weight.dtype)
        out.left.weight.copy_(a)
        out.right.weight.copy_(b)
        if layer.bias is not None:
            out.left.bias.copy_(layer.bias)
        return out


def select(model, policy: dict) -> dict[str, int]:
    """Explicit Qwen3 linear targets; no embeddings, fused/hybrid or tied weights."""
    if model.config.model_type != "qwen3":
        raise ValueError("only inspected Qwen3ForCausalLM is supported; audit other architectures first")
    kind = policy.get("kind", "mlp")
    endings = {"mlp": r"mlp\.(gate_proj|up_proj|down_proj)",
               "down": r"mlp\.down_proj", "attention": r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"}
    if kind not in endings:
        raise ValueError(f"unknown layer policy: {kind}")
    regex = re.compile(r"model\.layers\.(\d+)\." + endings[kind] + r"$")
    aliases = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    ranks = {}
    layer_ids = policy.get("layer_ids")
    if layer_ids is not None and (not layer_ids or len(set(layer_ids)) != len(layer_ids)):
        raise ValueError("layer_ids must be unique and nonempty")
    seen = set()
    for name, layer in model.named_modules():
        match = regex.fullmatch(name)
        if not match or (layer_ids is not None and int(match[1]) not in layer_ids):
            continue
        if type(layer) is not nn.Linear or aliases[id(layer.weight)] != 1:
            raise ValueError(f"unsupported or shared target: {name}")
        r = policy.get("ranks", {}).get(name, rank_for(layer.weight.shape, policy["keep"]))
        validate_rank(*layer.weight.shape, r)
        ranks[name] = r
        seen.add(int(match[1]))
    if not ranks or (layer_ids is not None and seen != set(layer_ids)):
        raise ValueError("empty selection or unmatched layer IDs")
    if set(policy.get("ranks", {})) - set(ranks):
        raise ValueError("rank override names do not match selected layers")
    return ranks


def replace(model, ranks: dict[str, int], *, initialize=True, scales=None):
    for name, rank in ranks.items():
        old = model.get_submodule(name)
        if type(old) is not nn.Linear:
            raise ValueError(f"expected dense nn.Linear at {name}")
        new = (FactorizedLinear.from_linear(old, rank, (scales or {}).get(name)) if initialize
               else FactorizedLinear(old.in_features, old.out_features, rank, old.bias is not None,
                                     device=old.weight.device, dtype=old.weight.dtype))
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, new)
    return model


class Constraint:
    def __init__(self, model, ranks):
        self.model, self.ranks, self.proxies = model, dict(ranks), {}

    @torch.no_grad()
    def refresh(self):
        # Sequential workspaces; detach is essential for the alternating update.
        for name, rank in self.ranks.items():
            self.proxies[name] = project(self.model.get_submodule(name).weight, rank)

    def penalty(self, mu):
        if mu <= 0 or set(self.proxies) != set(self.ranks):
            raise ValueError("positive mu and a complete proxy refresh are required")
        return sum((self.model.get_submodule(n).weight.float() - v).square().sum()
                   for n, v in self.proxies.items()) / (2 * mu)

    def state_dict(self):
        return {"ranks": self.ranks, "proxies": {k: v.cpu() for k, v in self.proxies.items()}}

    def load_state_dict(self, state):
        if state["ranks"] != self.ranks:
            raise ValueError("checkpoint target ranks differ")
        self.proxies = {}
        for n, value in state["proxies"].items():
            w = self.model.get_submodule(n).weight
            if value.shape != w.shape:
                raise ValueError("proxy shape differs")
            self.proxies[n] = value.to(w.device).detach()


def accounting(model, ranks):
    whole = sum(p.numel() for p in model.parameters())  # tied parameters counted once
    dense, compressed, biases, detail = 0, 0, 0, []
    for name, rank in ranks.items():
        layer = model.get_submodule(name)
        m, n = layer.out_features, layer.in_features
        bias = (layer.left.bias if isinstance(layer, FactorizedLinear) else layer.bias)
        b = 0 if bias is None else bias.numel()
        dense += m*n
        compressed += rank*(m+n)
        biases += b
        detail.append({"name": name, "shape": [m, n], "rank": rank,
                       "dense_weights": m*n, "factor_weights": rank*(m+n), "bias": b})
    is_fac = bool(ranks) and all(isinstance(model.get_submodule(n), FactorizedLinear) for n in ranks)
    original = whole + dense - compressed if is_fac else whole
    deployed = whole if is_fac else whole - dense + compressed
    return {"original_parameters": original, "deployment_parameters": deployed,
            "whole_keep": deployed/original, "selected_dense_with_bias": dense+biases,
            "selected_factors_with_bias": compressed+biases,
            "selected_keep": (compressed+biases)/(dense+biases) if dense else 1.0,
            "layers": detail}
