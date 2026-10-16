import pytest
from refldyn.metrics import summarize, cluster_bootstrap
from refldyn.analyze import cross_checkpoint_matrix, match_checkpoints


def row(i, checkpoint="base", p=0.8, initial=False, outcome=True):
    return {"id": f"p{i}-0", "problem_id": f"p{i}", "split": "validation", "checkpoint": checkpoint,
            "initial_correct": initial, "p_reflect": p, "natural_action": "REFLECT",
            "natural_correct": outcome, "natural_valid": True,
            "reflect_samples": [{"tokens": 120, "correct": outcome, "valid": True},
                                {"tokens": 100, "correct": outcome, "valid": True}]}


def test_selectivity_nontrivial():
    records = [row(0, p=1, initial=False, outcome=True), row(1, p=0, initial=True, outcome=False)]
    result = summarize(records)
    assert result["selectivity"] > 0
    assert result["repair_rate"] == 1
    assert result["damage_rate"] == 1
    intervals = cluster_bootstrap(records, bootstrap=20)
    assert "selectivity" in intervals


def test_cross_checkpoint_swaps():
    values = [row(0, "base", 0.1), row(1, "base", 0.1), row(0, "step40", 0.9), row(1, "step40", 0.9)]
    matrix = cross_checkpoint_matrix(values)
    assert len(matrix) == 4
    assert len({x["expected_gain"] for x in matrix}) > 1


def test_different_panels_rejected():
    groups = {("test", "step1"): [row(0)], ("test", "step2"): [row(1, "step2")]}
    with pytest.raises(ValueError):
        match_checkpoints(groups)
