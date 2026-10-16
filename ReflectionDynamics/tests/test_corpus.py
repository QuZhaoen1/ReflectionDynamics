import pytest
from refldyn.corpus import adapt_math, build_corpus, designate


def fixture(question, gold="7", source="math", split="train"):
    return {"question": question, "gold": gold, "source": source, "source_split": split}


def test_adapt_math():
    result = adapt_math({"problem": "Find x", "solution": r"\boxed{4}", "level": "Level 5", "type": "Algebra"}, "train")
    assert result["gold"] == "4"
    assert result["level"] == "Level 5"


def test_preserve_original_test_and_ood():
    assert designate("math", "test", "Problem X") == "test"
    assert designate("aime", "benchmark", "Problem Y") == "ood"


def test_overlapping_test_removes_train():
    corpus, rejected, report = build_corpus([fixture("What is two?", "2", split="train"),
                                              fixture("What   is TWO?", "2", split="test")])
    assert len(corpus) == 1
    assert corpus[0]["split"] == "test"
    assert report["accepted"] == 1


def test_bad_reference_rejected():
    with pytest.raises(ValueError):
        build_corpus([fixture("What is x?", "")])
