from refldyn.protocol import prompt_for_state, parse_decision, terminal_score


ROW = {"question": "Compute 1+1", "initial_reasoning": "1+1 is 3", "initial_answer": "3", "gold": "2"}


def test_prompt_does_not_leak_gold():
    prompt = prompt_for_state(ROW)
    assert "Compute 1+1" in prompt
    assert "Proposed answer: \\boxed{3}" in prompt
    assert "gold" not in prompt.lower()
    assert "\\boxed{2}" not in prompt


def test_parse_and_reward():
    good = "REFLECT\nCHECK: 1+1=2\nFINAL: \\boxed{2}"
    assert parse_decision(good, "3").valid
    assert terminal_score(good, "3", "2") == 1
    bad = "COMMIT\nFINAL: \\boxed{2}"
    assert not parse_decision(bad, "3").valid
    assert terminal_score(bad, "3", "2") == 0


def test_commit():
    assert parse_decision("COMMIT\nFINAL: \\boxed{3}", "3").valid
    assert not parse_decision("REFLECT\nFINAL: \\boxed{2}", "3").valid
