import pytest

from refldyn.curation import categorize, curate_training


def inputs(n=80):
    states = []
    evals = []
    for i in range(n):
        kind = i % 4
        correct = kind in (2, 3)
        state = {"id": f"p{i}-s0", "problem_id": f"p{i}", "split": "train",
                 "source": "math", "draft_model": "Qwen/Qwen2.5-32B",
                 "initial_correct": correct, "initial_answer": "2", "initial_reasoning": "reason",
                 "gold": "3", "question": f"Question {i}"}
        reflected = [True, True, False, False] if kind == 0 else (
                    [False, False, False, False] if kind == 1 else (
                    [True, True, True, False] if kind == 2 else [True] * 4))
        result = {"id": state["id"], "problem_id": state["problem_id"], "split": "train",
                  "checkpoint": "base", "reflect_samples": [
                      {"valid": True, "correct": x, "tokens": 100} for x in reflected]}
        states.append(state)
        evals.append(result)
    return states, evals


def test_four_training_cases_are_distinct():
    states, evals = inputs()
    selected, report = curate_training(states, evals, target=60, seed=2)
    assert len(selected) == 60
    assert len({r["problem_id"] for r in selected}) == 60
    assert set(report["selected_categories"]) == {
        "recoverable_error", "unresolved_error", "fragile_correct", "stable_correct"}
    assert all(r["curated_with"] == "base_forced_reflection" for r in selected)


def test_reproducible_selection():
    states, evals = inputs()
    one, report = curate_training(states, evals, target=40)
    two, _ = curate_training(states, evals, target=40)
    assert [x["id"] for x in one] == [x["id"] for x in two]


def test_curation_refuses_validation_and_model_checkpoints():
    states, evals = inputs(4)
    evals[0]["checkpoint"] = "step100"
    with pytest.raises(ValueError, match="pre-training"):
        curate_training(states, evals)
    evals[0]["checkpoint"] = "base"
    states[0]["split"] = "test"
    with pytest.raises(ValueError, match="training"):
        curate_training(states, evals)


def test_probe_missing_does_not_fabricate_labels():
    states, evals = inputs(12)
    chosen, report = curate_training(states, evals[:8], target=10)
    assert len(chosen) == 8
    assert report["discarded"]["unprobed"] == 4


def test_invalid_probes_are_refused():
    states, evals = inputs(4)
    for item in evals[0]["reflect_samples"][:2]:
        item["valid"] = False
    result = categorize(states[0], evals[0])
    assert result["reason"] == "low_reflection_parse_rate"
