import json
import sys

import pytest

from refldyn import evaluate, rollouts
from refldyn.io import load_jsonl, save_jsonl
from test_eval_backend import FakeBackend


class DraftBackend(FakeBackend):
    def __init__(self, *args, **kwargs):
        pass


def test_draft_can_extend_without_resampling(monkeypatch, tmp_path):
    dataset = tmp_path / "questions.jsonl"
    save_jsonl(dataset, [{"id": f"p{i}", "split": "train", "source": "math",
                          "question": f"Compute {i} - {i} + 7", "gold": "7", "level": "Level 5"}
                         for i in range(8)])
    dest = tmp_path / "drafts.jsonl"
    monkeypatch.setattr(rollouts, "InferenceBackend", DraftBackend)
    def execute(limit, resume=False):
        argv = ["draft", "--questions", str(dataset), "--output", str(dest),
                "--samples", "2", "--limit", str(limit),
                "--model", "Qwen/Qwen2.5-32B", "--max-new-tokens", "128"]
        if resume:
            argv.append("--resume")
        monkeypatch.setattr(sys, "argv", argv)
        rollouts.run()
    execute(2)
    original = {(r["id"], r["sample"]): r for r in load_jsonl(dest)}
    assert len(original) == 4
    execute(5, resume=True)
    extended = {(r["id"], r["sample"]): r for r in load_jsonl(dest)}
    assert len(extended) == 10
    assert all(extended[k] == v for k, v in original.items())
    assert json.loads((tmp_path / "drafts.jsonl.manifest.json").read_text())["limit"] == 5
    with pytest.raises(ValueError, match="shrink"):
        execute(3, resume=True)


def test_evaluation_resume_preserves_fixed_panel(monkeypatch, tmp_path):
    dataset = tmp_path / "states.jsonl"
    save_jsonl(dataset, [{"id": f"p{i}-s0", "problem_id": f"p{i}", "split": "validation",
                          "source": "math", "question": "3+4", "gold": "7",
                          "initial_answer": "8", "initial_reasoning": "3+4=8", "initial_correct": False}
                         for i in range(5)])
    dest = tmp_path / "evaluations.jsonl"
    monkeypatch.setattr(evaluate, "InferenceBackend", DraftBackend)
    def execute(resume=False):
        argv = ["evaluate", "--states", str(dataset), "--output", str(dest),
                "--checkpoint", "base", "--model", "Qwen/Qwen2.5-32B",
                "--repetitions", "2", "--retry-control"]
        if resume:
            argv.append("--resume")
        monkeypatch.setattr(sys, "argv", argv)
        evaluate.run()
    execute()
    originals = list(load_jsonl(dest))
    assert len(originals) == 5
    execute(True)
    assert list(load_jsonl(dest)) == originals
