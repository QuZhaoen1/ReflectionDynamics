from refldyn.rollouts import collect_draft, draft_prompt
from refldyn.evaluate import evaluate_state


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.encode("utf-8")


class FakeBackend:
    tokenizer = FakeTokenizer()

    def action_probability(self, text):
        return 0.75, [-2.1, -0.4]

    def generate(self, prompt, seed, max_new_tokens, sample=True):
        if prompt.endswith("ACTION: REFLECT\nCHECK: "):
            return "checked\nFINAL: \\boxed{7}", 12
        if "ACTION: " in prompt:
            return "REFLECT\nCHECK: checked\nFINAL: \\boxed{7}", 13
        return "Calculated. Answer is \\boxed{7}.", 14


def test_real_draft_data_format():
    row = {"id": "x", "split": "train", "source": "math", "question": "3+4", "gold": "7"}
    out = collect_draft(FakeBackend(), row, 41, 100)
    assert out["valid"] and out["correct"]
    assert out["initial_answer"] == "7"


def test_reflection_evaluation():
    row = {"id": "x-s0", "problem_id": "x", "split": "validation", "source": "math",
           "question": "3+4", "gold": "7", "initial_reasoning": "3+4=8", "initial_answer": "8",
           "initial_correct": False}
    out = evaluate_state(FakeBackend(), row, "base", 41, 2, 100, retry=True)
    assert out["natural_correct"]
    assert out["reflect_samples"][0]["correct"]
    assert out["retry_samples"][1]["correct"]
    assert out["p_reflect"] == 0.75


def test_long_prompt_skipped_without_crash():
    row = {"id": "x", "split": "train", "source": "math", "question": "big", "gold": "2"}
    out = collect_draft(FakeBackend(), row, 41, 100, prompt_limit=5)
    assert not out["valid"] and out["split"] == "train"
