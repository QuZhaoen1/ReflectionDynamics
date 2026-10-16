import pytest
from refldyn.selection import build_states, ensure_isolation
from refldyn.io import save_jsonl


def drafts(split="train"):
    result = []
    for item, correct in enumerate([True, False, True, False]):
        result.append({"id": "p1", "split": split, "sample": item, "question": "3+4?", "gold": "7",
                       "source": "math", "model": "Qwen/Qwen2.5-32B", "initial_answer": "7" if correct else "8",
                       "trace": "work", "correct": correct, "valid": True})
    return result


def test_states_true_model_trajectories():
    states, audit, report = build_states(drafts(), "train", minimum_samples=4)
    assert report["states"] == 2
    assert {x["initial_correct"] for x in states} == {True, False}
    assert all(x["draft_model"] == "Qwen/Qwen2.5-32B" for x in states)


def test_wrong_split_rejected():
    with pytest.raises(ValueError):
        build_states(drafts("test"), "train")


def test_split_isolation(tmp_path):
    a, _audit, _report = build_states(drafts(), "train", minimum_samples=4)
    save_jsonl(tmp_path / "train.jsonl", a)
    assert ensure_isolation({"train": tmp_path / "train.jsonl"})["unique_problems"] == 1
    save_jsonl(tmp_path / "test.jsonl", [{**x, "split": "test"} for x in a])
    with pytest.raises(ValueError, match="Leakage"):
        ensure_isolation({"train": tmp_path / "train.jsonl", "test": tmp_path / "test.jsonl"})
