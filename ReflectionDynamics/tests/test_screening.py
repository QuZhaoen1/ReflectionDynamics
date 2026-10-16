import pytest

from refldyn.screening import screen_drafts, wilson_interval


def drafts(pid, split="train", sample_count=8, correct_count=4, invalid_count=0):
    return [{"id": pid, "sample": i, "split": split, "model": "Qwen/Qwen2.5-32B",
             "question": f"Problem {pid}", "gold": "4", "level": "Level 5", "topic": "Number Theory",
             "valid": i < sample_count - invalid_count,
             "correct": i < correct_count if i < sample_count - invalid_count else None}
            for i in range(sample_count)]


def test_wilson_interval_behaviour():
    left, right = wilson_interval(4, 8)
    assert left < 0.5 < right
    assert wilson_interval(0, 0) == [None, None]


def test_screen_difficulty_coverage():
    rows, summary = screen_drafts(drafts("mid", correct_count=4) +
                                  drafts("easy", correct_count=8) +
                                  drafts("hard", correct_count=0), "train")
    bins = {x["problem_id"]: x["difficulty_band"] for x in rows}
    assert bins == {"easy": "too_easy", "hard": "too_hard", "mid": "informative"}
    assert summary["eligible_problems"] == 3
    assert summary["observed_parse_rate"] == 1


def test_invalid_generation_not_hidden():
    rows, report = screen_drafts(drafts("x", correct_count=3, invalid_count=3), "train")
    assert not rows[0]["eligible"]
    assert rows[0]["difficulty_band"] == "insufficient_evidence"
    assert report["observed_parse_rate"] == 5 / 8


def test_mismatched_model_rejected():
    rows = drafts("x")
    rows[0]["model"] = "Qwen/Qwen2.5-7B"
    with pytest.raises(ValueError, match="Inconsistent"):
        screen_drafts(rows, "train")
