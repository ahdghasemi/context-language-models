import copy
import json

import pytest
import torch

from tiny_clm.io import read_json
from tiny_clm.smoke import prepare_tiny
from tiny_clm.train import train


def test_resume_matches_uninterrupted_across_stage(tmp_path):
    model, cfg, data = prepare_tiny(tmp_path/"assets")
    initial = copy.deepcopy(model)
    assert train(model, data["train"], data["validation"], cfg, tmp_path/"full")
    assert not train(initial, data["train"], data["validation"], cfg, tmp_path/"resume", stop_after=2)
    restart = copy.deepcopy(initial)
    assert train(restart, data["train"], data["validation"], cfg, tmp_path/"resume", resume=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restart.state_dict()[name], rtol=0, atol=0)
    a = torch.load(tmp_path/"full/latest.pt", weights_only=False)
    b = torch.load(tmp_path/"resume/latest.pt", weights_only=False)
    assert a["tokens"] == b["tokens"] == 6*2*15
    assert a["total_steps"] == b["total_steps"] == 6
    assert torch.equal(a["sampler"]["order"], b["sampler"]["order"])
    assert read_json(tmp_path/"resume/status.json")["status"] == "complete"


def test_resume_rejects_changed_config_and_collisions(tmp_path):
    model, cfg, data = prepare_tiny(tmp_path/"assets")
    assert not train(model, data["train"], data["validation"], cfg, tmp_path/"run", stop_after=1)
    with pytest.raises(FileExistsError):
        train(model, data["train"], data["validation"], cfg, tmp_path/"run")
    cfg["seed"] = 99
    with pytest.raises(ValueError, match="configuration"):
        train(model, data["train"], data["validation"], cfg, tmp_path/"run", resume=True)


@pytest.mark.parametrize("method", ["dense_matched", "svd_recovery", "asvd_uniform_recovery"])
def test_matched_training_tokens_and_factor_gradients(tmp_path, method):
    model, cfg, data = prepare_tiny(tmp_path/"assets")
    cfg["method"] = method
    assert train(model, data["train"], data["validation"], cfg, tmp_path/"run")
    state = torch.load(tmp_path/"run/latest.pt", weights_only=False)
    assert state["tokens"] == 180
    assert state["factorized"] == (method != "dense_matched")
    rows = [json.loads(s) for s in (tmp_path/"run/training.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and all(r["penalty"] == 0 for r in rows)


def test_bf16_export_projects_trained_fp32_weights_before_rounding(tmp_path):
    from tiny_clm.evaluate import export_run
    from tiny_clm.lowrank import factors, select
    from tiny_clm.model import load_export
    model, cfg, data = prepare_tiny(tmp_path/"assets")
    cfg["export_dtype"] = "bfloat16"
    assert train(model, data["train"], data["validation"], cfg, tmp_path/"run")
    ranks = select(model, cfg["selection"])
    expected = {name: factors(model.get_submodule(name).weight, rank) for name, rank in ranks.items()}
    export_run(tmp_path/"run")
    deployed, _ = load_export(tmp_path/"run/export")
    for name, (left, right) in expected.items():
        layer = deployed.get_submodule(name)
        torch.testing.assert_close(layer.left.weight, left.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(layer.right.weight, right.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("policy", ["gnn_persistent", "flow_reset"])
def test_optimizer_stage_semantics_and_restart(tmp_path, policy):
    model, cfg, data = prepare_tiny(tmp_path/"assets")
    cfg["training"]["optimizer_policy"] = policy
    reference = copy.deepcopy(model)
    assert not train(model, data["train"], data["validation"], cfg, tmp_path/"run", stop_after=4)
    state = torch.load(tmp_path/"run/latest.pt", weights_only=False)
    adam_steps = {int(v["step"]) for v in state["optimizer"]["state"].values()}
    assert adam_steps == ({4} if policy == "gnn_persistent" else {1})
    assert train(model, data["train"], data["validation"], cfg, tmp_path/"run", resume=True)
    assert train(reference, data["train"], data["validation"], cfg, tmp_path/"full")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)


def test_advance_signal_saves_incomplete_checkpoint(tmp_path):
    import os
    import signal
    model, cfg, data = prepare_tiny(tmp_path/"assets")
    fired = []

    def interrupt_once(*_):
        if not fired:
            fired.append(True)
            os.kill(os.getpid(), signal.SIGUSR1)

    handle = model.register_forward_pre_hook(interrupt_once)
    try:
        assert not train(model, data["train"], data["validation"], cfg, tmp_path/"run")
    finally:
        handle.remove()
    state = torch.load(tmp_path/"run/latest.pt", weights_only=False)
    assert not state["complete"] and state["total_steps"] == 1
    assert state["optimizer"] and state["scheduler"] and state["constraint"]
    assert train(model, data["train"], data["validation"], cfg, tmp_path/"run", resume=True)
