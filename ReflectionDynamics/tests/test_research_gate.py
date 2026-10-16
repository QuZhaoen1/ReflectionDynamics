import pytest

from refldyn.gates import build_gate
from refldyn.io import save_jsonl, checksum_file
from test_metrics import row as sample_row


def make_training(number):
    correct = number % 2 == 0
    return {"id": f"tr{number}-s0", "problem_id": f"tr{number}", "split": "train",
            "source": "math", "draft_model": "Qwen/Qwen2.5-32B",
            "initial_reasoning": "initial reasoning", "initial_answer": "3",
            "initial_correct": correct, "gold": "2", "question": f"Problem {number}",
            "curated_with": "base_forced_reflection",
            "probe_category": "stable_correct" if correct else "recoverable_error"}


def validation_record(number):
    base = sample_row(number, p=0.65, initial=False, outcome=True)
    base["id"] = f"val{number}-s0"
    base["problem_id"] = f"val{number}"
    return base


def setup_gate(tmp_path):
    training = [make_training(i) for i in range(320)]
    validation = [{"id": f"val{i}-s0", "problem_id": f"val{i}", "split": "validation"}
                  for i in range(45)]
    evaluation = [validation_record(i) for i in range(45)]
    path = tmp_path / "train.jsonl"
    save_jsonl(path, training)
    report = {"training_sha256": checksum_file(path), "curated_states": len(training)}
    return training, validation, evaluation, path, report


def test_full_research_gate_uses_data_provenance(tmp_path):
    training, validation, evaluation, path, provenance = setup_gate(tmp_path)
    gate = build_gate(training, evaluation, "Qwen/Qwen2.5-32B", path,
                      validation_states=validation, curation_report=provenance)
    assert gate["passed"]
    assert gate["validation_coverage"] == 1.0
    assert gate["curation_report_verified"]


def test_curation_hash_mismatch_stops_training(tmp_path):
    training, validation, evaluation, path, provenance = setup_gate(tmp_path)
    provenance["training_sha256"] = "fake-sha"
    gate = build_gate(training, evaluation, "Qwen/Qwen2.5-32B", path,
                      validation_states=validation, curation_report=provenance)
    assert not gate["passed"]
    assert not gate["conditions"]["trained_on_probed_states"]


def test_missing_baseline_panel_rejected(tmp_path):
    training, validation, evaluation, path, provenance = setup_gate(tmp_path)
    with pytest.raises(ValueError, match="fixed validation panel"):
        build_gate(training, evaluation[:-1], "Qwen/Qwen2.5-32B", path,
                   validation_states=validation, curation_report=provenance)


def test_train_and_validation_overlap_rejected(tmp_path):
    training, validation, evaluation, path, provenance = setup_gate(tmp_path)
    validation[0]["problem_id"] = training[0]["problem_id"]
    with pytest.raises(ValueError, match="leakage"):
        build_gate(training, evaluation, "Qwen/Qwen2.5-32B", path,
                   validation_states=validation, curation_report=provenance)
