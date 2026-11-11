import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tiny_clm.contextbench import tree_hash, verify_lock
from tiny_clm.io import digest


def test_task_lock_detects_tampering_and_changed_config(tmp_path, monkeypatch):
    import tiny_clm.contextbench as cb
    monkeypatch.setattr(cb, "check_source", lambda *_: None)
    task = tmp_path/"task"
    task.mkdir()
    (task/"task.toml").write_text("original")
    cache = tmp_path/"cache"
    cache.mkdir()
    (cache/"vocab").write_text("tokenizer")
    cfg = {"seeds": [1]}
    lock = {"request_hash": digest(cfg), "runnable": False, "source": "unused", "revision": "unused",
            "tasks": [{"path": str(task), "sha256": tree_hash(task)}],
            "budget_tokenizer_cache": {"path": str(cache), "sha256": tree_hash(cache)}}
    verify_lock(lock, cfg, runnable=False)
    with pytest.raises(ValueError):
        verify_lock(lock, cfg)
    with pytest.raises(ValueError):
        verify_lock(lock, {"seeds": [2]}, runnable=False)
    (task/"task.toml").write_text("changed")
    with pytest.raises(ValueError):
        verify_lock(lock, cfg, runnable=False)


def test_ablation_isolates_mu_with_same_updates_and_optimizer_resets():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("submit", root/"submit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((root/"configs/pilot.json").read_text())
    original = copy.deepcopy(cfg)
    rows = module.grid(cfg, "ablations")
    assert cfg == original and len(rows) == 9
    assert len({r["run_id"] for r in rows}) == 9
    controls = [r["config"] for r in rows if "constant_" in r["run_id"]]
    assert len(controls) == 2
    for row in controls:
        assert row["training"] == cfg["training"]
        assert row["penalty"] == cfg["penalty"]
        assert row["ablation_constant_mu"] in [100., 10000.]
