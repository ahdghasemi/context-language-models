"""Detect BF16 RoPE damage independently of weight serialization accuracy."""
import json

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from tiny_clm.lowrank import replace, select
from tiny_clm.model import deployment_precision, export_model, inference_state, load_export
from tiny_clm.provenance import INFERENCE_RUNTIME

torch.set_num_threads(2)


def source_model(dtype=torch.bfloat16):
    torch.manual_seed(123)
    config = Qwen3Config(vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=1, num_key_value_heads=1, head_dim=128,
        tie_word_embeddings=True, rope_theta=5000000.0, max_position_embeddings=262144,
        eos_token_id=2, bos_token_id=1, pad_token_id=0, attention_dropout=0.0)
    # Construct at the target weight dtype exactly as HF loading does; casting
    # the entire module here would round both sides and mask the original bug.
    return AutoModelForCausalLM.from_config(config, torch_dtype=dtype, attn_implementation="sdpa").eval()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("factorized", [False, True])
def test_native_precision_export_retains_weights_buffers_logits_and_tying(tmp_path, dtype, factorized):
    model = source_model(getattr(torch, dtype))
    ranks = select(model, {"kind": "mlp", "keep": 0.5}) if factorized else {}
    replace(model, ranks)
    ids = torch.arange(1, 65).reshape(1, -1)
    positions = torch.arange(4096, 4160).reshape(1, -1)
    before_state = inference_state(model)
    with torch.inference_mode():
        before = model(ids, position_ids=positions, use_cache=False).logits
    artifact = tmp_path/"export"
    report = export_model(model, artifact, ranks, dtype=dtype)
    loaded, _ = load_export(artifact)
    assert report["format"] == "tiny-clm-factorized-v2"
    assert report["inference_runtime"] == INFERENCE_RUNTIME
    assert inference_state(loaded) == before_state == inference_state(model)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert loaded.model.rotary_emb.inv_freq.dtype == torch.float32
    with torch.inference_mode():
        after = loaded(ids, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert all(n+".weight" not in loaded.state_dict() for n in ranks)


def test_identical_weights_do_not_imply_identical_runtime_outputs():
    model = source_model()
    state = inference_state(model)
    freq = model.model.rotary_emb.inv_freq.clone()
    ids = torch.arange(1, 65).reshape(1, -1)
    positions = torch.arange(4096, 4160).reshape(1, -1)
    with torch.inference_mode():
        before = model(ids, position_ids=positions, use_cache=False).logits.float()
        model.to(dtype=torch.bfloat16)  # Reproduce the historical bug.
        damaged = model(ids, position_ids=positions, use_cache=False).logits.float()
        changed = inference_state(model)
        model.model.rotary_emb.inv_freq = freq
        restored = model(ids, position_ids=positions, use_cache=False).logits.float()
    assert state["parameter_sha256"] == changed["parameter_sha256"]
    assert state["buffers"] != changed["buffers"]
    assert (before-damaged).abs().max().item() > 1e-4
    torch.testing.assert_close(restored, before, rtol=0, atol=0)


@pytest.mark.parametrize("upcast_again", [False, True])
def test_already_rounded_frequencies_are_rejected(tmp_path, upcast_again):
    model = source_model().to(dtype=torch.bfloat16)
    if upcast_again:
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.float()
    with pytest.raises(ValueError, match="rounded or changed"):
        export_model(model, tmp_path/"export", {}, dtype="bfloat16")
    assert not (tmp_path/"export").exists()


def test_weight_conversion_preserves_full_precision_frequency_values():
    model = source_model(torch.float32)
    freq = model.model.rotary_emb.inv_freq.clone()
    deployment_precision(model, "bfloat16")
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, freq, rtol=0, atol=0)
    assert model.model.rotary_emb.original_inv_freq is model.model.rotary_emb.inv_freq
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_legacy_artifacts_keep_explicit_legacy_behavior(tmp_path):
    model = source_model()
    export_model(model, tmp_path/"export", {}, dtype="bfloat16")
    path = tmp_path/"export/compression.json"
    spec = json.loads(path.read_text())
    spec["format"] = "tiny-clm-factorized-v1"
    del spec["inference_runtime"]
    path.write_text(json.dumps(spec))
    with pytest.warns(RuntimeWarning, match="Legacy v1 artifact"):
        legacy, _ = load_export(path.parent)
    assert legacy.model.rotary_emb.inv_freq.dtype == torch.bfloat16
    assert inference_state(legacy)["parameter_sha256"] == inference_state(model)["parameter_sha256"]


def test_unknown_runtime_cannot_silently_load(tmp_path):
    export_model(source_model(), tmp_path/"export", {}, dtype="bfloat16")
    path = tmp_path/"export/compression.json"
    spec = json.loads(path.read_text())
    spec["inference_runtime"] = "unreviewed"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="precision policy"):
        load_export(path.parent)


def test_bf16_generation_and_cached_execution_after_reload(tmp_path):
    model = source_model()
    export_model(model, tmp_path/"export", {}, dtype="bfloat16")
    loaded, _ = load_export(tmp_path/"export")
    ids = torch.tensor([[1, 4, 5, 6]])
    with torch.inference_mode():
        full = loaded(ids, use_cache=False).logits[:, -1].float()
        prefix = loaded(ids[:, :-1], use_cache=True)
        cached = loaded(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1].float()
        torch.testing.assert_close(cached, full, rtol=0.02, atol=0.25)
        original_tokens = model.generate(ids, max_new_tokens=3, do_sample=False)
        reloaded_tokens = loaded.generate(ids, max_new_tokens=3, do_sample=False)
    assert torch.equal(original_tokens, reloaded_tokens)


def test_scaled_rope_is_not_silently_treated_as_default():
    model = source_model()
    model.model.rotary_emb.rope_type = "dynamic"
    with pytest.raises(ValueError, match="scaled/dynamic"):
        deployment_precision(model, "bfloat16")
