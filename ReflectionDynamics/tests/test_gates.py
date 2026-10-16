import pytest
from refldyn.gates import assess_train, build_gate, verify_gate
from refldyn.io import save_jsonl, write_json
from test_metrics import row


def state(number, split="train", source="math"):
    return {"id": f"p{number}-s0", "problem_id": f"p{number}", "split": split,
            "source": source, "draft_model": "Qwen/Qwen2.5-32B", "initial_reasoning": "work",
            "initial_answer": "8", "initial_correct": number % 2 == 0, "gold": "7", "question": "What is 3+4?"}


def test_must_be_real_training_states():
    with pytest.raises(ValueError):
        assess_train([state(0, source="synthetic_integer_arithmetic")], "Qwen/Qwen2.5-32B")
    with pytest.raises(ValueError):
        assess_train([state(0, split="test")], "Qwen/Qwen2.5-32B")


def test_gate_pass_and_binding(tmp_path):
    items = [state(x) for x in range(320)]
    path = tmp_path / "train.jsonl"
    save_jsonl(path, items)
    evaluation = [row(i, p=0.5, initial=False, outcome=True) for i in range(45)]
    output = build_gate(items, evaluation, "Qwen/Qwen2.5-32B", path)
    assert output["passed"]
    gate = tmp_path / "gate.json"
    write_json(gate, output)
    assert verify_gate(gate, "Qwen/Qwen2.5-32B", path)
    save_jsonl(path, items[:-1])
    with pytest.raises(ValueError, match="changed"):
        verify_gate(gate, "Qwen/Qwen2.5-32B", path)


def test_gate_insufficient_results(tmp_path):
    path = tmp_path / "train.jsonl"
    save_jsonl(path, [state(i) for i in range(320)])
    report = build_gate(list(__import__("refldyn.io", fromlist=["load_jsonl"]).load_jsonl(path)),
                        [row(0)], "Qwen/Qwen2.5-32B", path)
    assert not report["passed"]


def test_preview_fixture_forbidden():
    record = state(0)
    record["fixture"] = True
    with pytest.raises(ValueError):
        assess_train([record], "Qwen/Qwen2.5-32B", True)
