import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_dependencies_and_unique_grid():
    submit = load_script("submit")
    config = json.loads((ROOT/"configs/pilot.json").read_text())
    rows = submit.grid(config, "pilot")
    assert len({r["run_id"] for r in rows}) == len(rows) == 10
    plans = {p[0]: p for p in submit.make_plan("pilot", len(rows))}
    assert plans["train"][3] == [("afterok", "smoke")]
    assert plans["export"][3] == [("aftercorr", "train")]
    assert plans["evaluate"][3] == [("aftercorr", "export")]
    assert plans["profile"][3] == [("aftercorr", "export")]
    assert plans["contextbench"][3] == [("aftercorr", "export")]
    assert all(d[0] == "afterany" for d in plans["aggregate"][3])
    assert submit.make_plan("prepare", 0) == [("prepare", False, None, [])]


def test_aggregate_keeps_missing_runs_and_gate_fails(tmp_path):
    module = load_script("aggregate")
    config = json.loads((ROOT/"configs/pilot.json").read_text())
    spec = {"campaign": str(tmp_path), "config": config, "commit": "test",
            "rows": [{"run_id": "absent", "config": config}]}
    rows, episodes = module.collect(spec)
    assert len(rows) == 1 and rows[0]["status"] == "incomplete"
    assert "evaluation.json" in rows[0]["missing"]
    assert not module.gate(spec, rows, episodes)["passed"]


def test_single_gpu_resource_contract(monkeypatch):
    submit = load_script("submit")
    monkeypatch.setenv("TINY_CLM_GPUS", "2")
    with pytest.raises(ValueError):
        submit.resource_args(True, "train")
