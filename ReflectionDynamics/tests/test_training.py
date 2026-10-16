import pytest
from refldyn.train_grpo import prepare_training_records, outcome_reward
from refldyn.checks import inspect_config
from test_gates import state


def test_outcome_reward():
    result = outcome_reward(["REFLECT\nCHECK: recomputed\nFINAL: \\boxed{7}",
                             "COMMIT\nFINAL: \\boxed{8}"], ["8", "8"], ["7", "7"])
    assert result == [1, 0]


def test_prepare_split_guard():
    with pytest.raises(ValueError):
        prepare_training_records([state(0, split="test")], "Qwen/Qwen2.5-32B", True)


def test_training_profile():
    result = inspect_config("configs/grpo_32b_a10040.yaml")
    assert result["zero_stage"] == 3
    assert result["global_batch"] == 32


def test_model_draft_consistency():
    wrong = state(1)
    wrong["draft_model"] = "Qwen/Qwen2.5-7B"
    with pytest.raises(ValueError, match="Draft model"):
        prepare_training_records([wrong], "Qwen/Qwen2.5-32B", True)
