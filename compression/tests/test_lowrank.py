import copy

import pytest
import torch
from torch import nn

from tiny_clm.lowrank import (Constraint, FactorizedLinear, accounting, factors,
                             mu_schedule, project, rank_for, replace, select)
from tiny_clm.model import export_model, load_export
from tiny_clm.smoke import tiny_model

torch.set_num_threads(2)


def test_exact_hard_projection_and_tail_error():
    torch.manual_seed(3)
    w = torch.randn(12, 9, dtype=torch.float64)
    v = project(w, 3)
    s = torch.linalg.svdvals(w)
    assert torch.linalg.matrix_rank(v, atol=1e-10).item() == 3
    torch.testing.assert_close((w-v).square().sum(), s[3:].square().sum(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(project(v, 3), v)
    assert not v.requires_grad


def test_penalty_gradient_is_w_minus_proxy_over_mu():
    m = tiny_model()
    c = Constraint(m, {"model.layers.0.mlp.down_proj": 4})
    c.refresh()
    w = m.get_submodule("model.layers.0.mlp.down_proj").weight
    c.penalty(0.2).backward()
    torch.testing.assert_close(w.grad, (w.detach()-c.proxies["model.layers.0.mlp.down_proj"])/0.2)
    assert all(not v.requires_grad for v in c.proxies.values())


def test_schedule_direction_scaling():
    assert mu_schedule(8, 1, 0.5) == [8, 4, 2, 1]
    assert mu_schedule(10, 1, 0.5) == [10, 5, 2.5, 1.25]
    for bad in [(1, 2, 0.5), (1, 1, 1), (1, 0, 0.5)]:
        with pytest.raises(ValueError):
            mu_schedule(*bad)


@pytest.mark.parametrize("shape", [(12, 9), (9, 12)])
def test_factorized_linear_shapes_gradients_and_equivalence(shape):
    m, n = shape
    dense = nn.Linear(n, m, dtype=torch.float64)
    fac = FactorizedLinear.from_linear(dense, 3)
    x = torch.randn(2, 5, n, dtype=torch.float64, requires_grad=True)
    expected = nn.functional.linear(x, fac.left.weight@fac.right.weight, fac.left.bias)
    torch.testing.assert_close(fac(x), expected, rtol=1e-10, atol=1e-10)
    fac(x).sum().backward()
    assert x.grad is not None and all(p.grad is not None for p in fac.parameters())
    assert "weight" not in fac.state_dict()
    assert sum(p.numel() for p in fac.parameters()) == 3*(m+n)+m


def test_noncompressive_rank_and_invalid_layer_ids_rejected():
    with pytest.raises(ValueError):
        FactorizedLinear(8, 8, 4)
    with pytest.raises(ValueError):
        rank_for((4, 4), 0.01)
    with pytest.raises(ValueError):
        select(tiny_model(), {"kind": "mlp", "keep": .5, "layer_ids": [50]})


@pytest.mark.parametrize("policy", ["mlp", "attention"])
def test_qwen_tying_accounting_replacement_and_cache(tmp_path, policy):
    torch.manual_seed(0)
    m = tiny_model().eval()
    assert m.lm_head.weight is m.model.embed_tokens.weight
    before = sum(p.numel() for p in m.parameters())
    ranks = select(m, {"kind": policy, "keep": .5})
    counts = accounting(m, ranks)
    dense = copy.deepcopy(m)
    for name, rank in ranks.items():
        dense.get_submodule(name).weight.data.copy_(project(dense.get_submodule(name).weight, rank))
    replace(m, ranks)
    assert sum(p.numel() for p in m.parameters()) == counts["deployment_parameters"] < before
    assert m.lm_head.weight is m.model.embed_tokens.weight
    x = torch.tensor([[1, 5, 8, 4, 9]])
    with torch.no_grad():
        expected = dense(x).logits
        torch.testing.assert_close(m(x).logits, expected, rtol=2e-5, atol=2e-6)
        prefix = m(x[:, :-1], use_cache=True)
        cached = m(x[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits
        torch.testing.assert_close(cached[:, -1], m(x, use_cache=False).logits[:, -1], rtol=2e-5, atol=2e-6)
        g1 = m.generate(x, do_sample=False, max_new_tokens=3, use_cache=True)
        g2 = m.generate(x, do_sample=False, max_new_tokens=3, use_cache=False)
        assert torch.equal(g1, g2)
    export_model(m, tmp_path/"export", ranks, dtype="float32")
    loaded, spec = load_export(tmp_path/"export")
    torch.testing.assert_close(loaded(x).logits, expected, rtol=2e-5, atol=2e-6)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert accounting(loaded, ranks)["original_parameters"] == before
    assert all(name+".weight" not in loaded.state_dict() for name in ranks)


def test_asvd_unscales_and_does_not_enter_constrained_projection():
    w = torch.randn(12, 9)
    scale = torch.rand(9)+0.5
    a, b = factors(w, 3, scale)
    torch.testing.assert_close((a@b)*scale, project(w*scale, 3))



def test_export_rejects_dense_weights_labeled_as_factors(tmp_path):
    from tiny_clm.model import export_model
    from tiny_clm.smoke import tiny_model
    from tiny_clm.lowrank import select
    model = tiny_model()
    ranks = select(model, {"kind": "mlp", "keep": .5})
    with pytest.raises(ValueError, match="physical factors"):
        export_model(model, tmp_path/"export", ranks)


def test_explicit_ranks_do_not_require_keep_and_invalid_layer_ids_fail():
    from tiny_clm.smoke import tiny_model
    from tiny_clm.lowrank import select
    model = tiny_model()
    ranks = select(model, {"kind": "down", "layer_ids": [0], "keep": .5})
    assert select(model, {"kind": "down", "layer_ids": [0], "ranks": ranks}) == ranks
    with pytest.raises(ValueError):
        select(model, {"kind": "down", "layer_ids": [True], "keep": .5})
