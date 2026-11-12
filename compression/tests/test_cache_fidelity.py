"""Native cache fidelity is distinct from cached/full numerical equivalence."""
import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from tiny_clm.model import export_model, load_export
from tiny_clm.screen import cache_probe, verify_cache_probe
from tiny_clm.screen_contract import CACHE_CASES, cache_verification_passed

torch.set_num_threads(2)


def fixture():
    full = torch.tensor([[1., 2., 3.]])
    cached = torch.tensor([[1.5, 2.5, 3.5]])
    values = {c: {"full": full.clone(), "implicit": cached.clone(), "explicit": cached.clone()} for c in CACHE_CASES}
    tokens = {"cached": [1, 2, 3], "uncached": [1, 2, 4]}
    return values, copy.deepcopy(values), tokens, copy.deepcopy(tokens)


def test_native_rounding_and_cached_uncached_token_differences_stay_visible():
    check = verify_cache_probe(*fixture())
    assert check["passed"] and cache_verification_passed(check)
    for label in ("native", "exported"):
        assert all(not d["implicit_vs_full"]["allclose"]
                   for d in check["cached_vs_full_diagnostics"][label].values())
    assert check["generation_ids"]["native"]["cached"] != check["generation_ids"]["native"]["uncached"]


@pytest.mark.parametrize("fault", ["cached_logits", "full_logits", "explicit_positions", "cached_tokens", "uncached_tokens", "nan"])
def test_fidelity_faults_are_rejected(fault):
    native, exported, native_tokens, exported_tokens = fixture()
    if fault == "cached_logits":
        exported[CACHE_CASES[0]]["implicit"][0, 0] += .01  # Well below old .25 tolerance.
    elif fault == "full_logits":
        exported[CACHE_CASES[0]]["full"][0, 0] += .01
    elif fault == "explicit_positions":
        # Even a shared position discrepancy must not be mistaken for verified positions.
        for data in (native, exported):
            data[CACHE_CASES[0]]["explicit"][0, 0] += .01
    elif fault in {"cached_tokens", "uncached_tokens"}:
        exported_tokens[fault.split("_")[0]][0] += 1
    else:
        exported[CACHE_CASES[0]]["implicit"][0, 0] = float("nan")
    check = verify_cache_probe(native, exported, native_tokens, exported_tokens)
    assert not check["passed"] and not cache_verification_passed(check)


@pytest.mark.parametrize("fault", ["legacy", "missing_case", "loose_tolerance", "missing_generation"])
def test_incomplete_or_relabelled_evidence_cannot_pass(fault):
    check = verify_cache_probe(*fixture())
    if fault == "legacy":
        check = {"cache_passed": True}
    elif fault == "missing_case":
        del check["native_export"][CACHE_CASES[-1]]
    elif fault == "loose_tolerance":
        check["native_export"][CACHE_CASES[0]]["implicit"]["atol"] = .25
    else:
        del check["generation_equal"]
    assert not cache_verification_passed(check)


def test_real_bf16_screen_probe_round_trip(tmp_path):
    torch.manual_seed(4)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
        eos_token_id=2, pad_token_id=0, bos_token_id=1, max_position_embeddings=4096)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    ids = torch.arange(1, 25).reshape(1, -1)
    native, native_tokens = cache_probe(model, ids, 0)
    export_model(model, tmp_path/"export", {}, dtype="bfloat16")
    reloaded, _ = load_export(tmp_path/"export")
    exported, exported_tokens = cache_probe(reloaded, ids, 0)
    check = verify_cache_probe(native, exported, native_tokens, exported_tokens)
    assert check["passed"]
    assert native[CACHE_CASES[0]]["implicit"].shape == (1, 64)
    assert native[CACHE_CASES[2]]["implicit"].shape == (8, 64)
