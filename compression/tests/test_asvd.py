from tiny_clm.asvd import allocate, sensitivity_ranks
from tiny_clm.train import calibrate_asvd
from tiny_clm.smoke import prepare_tiny
from tiny_clm.lowrank import select


def test_exact_budget_allocation():
    shapes = {"a": (100, 100), "b": (100, 100)}
    ranks, fraction = allocate(shapes, [(1., "a", 20), (2., "b", 20)], .5)
    assert ranks == {"a": 20, "b": 20} and fraction == .4


def test_sensitivity_restores_dense_model(tmp_path):
    model, cfg, data = prepare_tiny(tmp_path)
    ranks = select(model, cfg["selection"])
    before = set(model.state_dict())
    scales = calibrate_asvd(model, ranks, data["train"], 1)
    chosen, report = sensitivity_ranks(model, ranks, scales, data["train"], 1, .6, candidates=(.4,.6,.8))
    assert set(model.state_dict()) == before
    assert report["actual_selected_keep"] <= .6
    assert chosen and len(report["records"]) == len(ranks)*3
