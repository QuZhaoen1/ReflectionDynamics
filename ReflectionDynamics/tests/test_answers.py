from refldyn.answers import last_boxed, gold_from_math, gold_from_gsm8k, verify_answer, extract_final


def test_boxed_nested():
    assert last_boxed(r"x=\\boxed{3}; y=\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_final("nothing") is None


def test_gsm_reference():
    assert gold_from_gsm8k("work\n#### 1,234") == "1234"
    assert gold_from_gsm8k("missing") is None


def test_numeric_verifier():
    assert verify_answer("1/2", "0.50") is True
    assert verify_answer("-3", "-2") is False
    assert verify_answer("1/0", "1") is None
    assert verify_answer("12", None) is None


def test_extract_solution():
    assert gold_from_math(r"finished with \boxed{17}") == "17"
    assert gold_from_math("solution", "19") == "19"
