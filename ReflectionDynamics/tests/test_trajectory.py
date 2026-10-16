import numpy as np
import pytest
from refldyn.trajectory import checkpoint_step, fit_transition


def test_checkpoint_labels():
    assert checkpoint_step("base") == 0
    assert checkpoint_step("step040") == 40
    assert checkpoint_step("checkpoint-120") == 120
    assert checkpoint_step("other") is None


def test_exploratory_transition():
    x = np.arange(0, 100, 10)
    y = np.sin(x / 45) / 3
    result = fit_transition(x, y)
    assert result["segmented_bic"] < 100
    assert "not evidence" in result["interpretation"]


def test_no_single_jump_claim():
    with pytest.raises(ValueError):
        fit_transition([1, 2, 3], [0.1, 0.2, 0.5])
