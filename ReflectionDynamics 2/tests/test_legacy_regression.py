from __future__ import annotations


# Existing regression tests grouped by their original module scopes.

_suites = {}


def _make_test_metrics():
    import pytest
    from refldyn.metrics import summarize, cluster_bootstrap
    from refldyn.analyze import cross_checkpoint_matrix, match_checkpoints


    def row(i, checkpoint="base", p=0.8, initial=False, outcome=True):
        return {"id": f"p{i}-0", "problem_id": f"p{i}", "split": "validation", "checkpoint": checkpoint,
                "initial_correct": initial, "p_reflect": p, "natural_action": "REFLECT",
                "natural_correct": outcome, "natural_valid": True,
                "reflect_samples": [{"tokens": 120, "correct": outcome, "valid": True},
                                    {"tokens": 100, "correct": outcome, "valid": True}]}


    def test_selectivity_nontrivial():
        records = [row(0, p=1, initial=False, outcome=True), row(1, p=0, initial=True, outcome=False)]
        result = summarize(records)
        assert result["selectivity"] > 0
        assert result["repair_rate"] == 1
        assert result["damage_rate"] == 1
        intervals = cluster_bootstrap(records, bootstrap=20)
        assert "selectivity" in intervals


    def test_cross_checkpoint_swaps():
        values = [row(0, "base", 0.1), row(1, "base", 0.1), row(0, "step40", 0.9), row(1, "step40", 0.9)]
        matrix = cross_checkpoint_matrix(values)
        assert len(matrix) == 4
        assert len({x["expected_gain"] for x in matrix}) > 1


    def test_different_panels_rejected():
        groups = {("test", "step1"): [row(0)], ("test", "step2"): [row(1, "step2")]}
        with pytest.raises(ValueError):
            match_checkpoints(groups)

    return locals()
_suites['test_metrics'] = _make_test_metrics()
for _key, _fn in _suites['test_metrics'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_metrics__{_key}"] = _fn
del _make_test_metrics


def _make_test_eval_backend():
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

    return locals()
_suites['test_eval_backend'] = _make_test_eval_backend()
for _key, _fn in _suites['test_eval_backend'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_eval_backend__{_key}"] = _fn
del _make_test_eval_backend


def _make_test_curation():
    import pytest

    from refldyn.curation import categorize, curate_training


    def inputs(n=80):
        states = []
        evals = []
        for i in range(n):
            kind = i % 4
            correct = kind in (2, 3)
            state = {"id": f"p{i}-s0", "problem_id": f"p{i}", "split": "train",
                     "source": "math", "draft_model": "Qwen/Qwen2.5-32B",
                     "initial_correct": correct, "initial_answer": "2", "initial_reasoning": "reason",
                     "gold": "3", "question": f"Question {i}"}
            reflected = [True, True, False, False] if kind == 0 else (
                        [False, False, False, False] if kind == 1 else (
                        [True, True, True, False] if kind == 2 else [True] * 4))
            result = {"id": state["id"], "problem_id": state["problem_id"], "split": "train",
                      "checkpoint": "base", "reflect_samples": [
                          {"valid": True, "correct": x, "tokens": 100} for x in reflected]}
            states.append(state)
            evals.append(result)
        return states, evals


    def test_four_training_cases_are_distinct():
        states, evals = inputs()
        selected, report = curate_training(states, evals, target=60, seed=2)
        assert len(selected) == 60
        assert len({r["problem_id"] for r in selected}) == 60
        assert set(report["selected_categories"]) == {
            "recoverable_error", "unresolved_error", "fragile_correct", "stable_correct"}
        assert all(r["curated_with"] == "base_forced_reflection" for r in selected)


    def test_reproducible_selection():
        states, evals = inputs()
        one, report = curate_training(states, evals, target=40)
        two, _ = curate_training(states, evals, target=40)
        assert [x["id"] for x in one] == [x["id"] for x in two]


    def test_curation_refuses_validation_and_model_checkpoints():
        states, evals = inputs(4)
        evals[0]["checkpoint"] = "step100"
        with pytest.raises(ValueError, match="pre-training"):
            curate_training(states, evals)
        evals[0]["checkpoint"] = "base"
        states[0]["split"] = "test"
        with pytest.raises(ValueError, match="training"):
            curate_training(states, evals)


    def test_probe_missing_does_not_fabricate_labels():
        states, evals = inputs(12)
        chosen, report = curate_training(states, evals[:8], target=10)
        assert len(chosen) == 8
        assert report["discarded"]["unprobed"] == 4


    def test_invalid_probes_are_refused():
        states, evals = inputs(4)
        for item in evals[0]["reflect_samples"][:2]:
            item["valid"] = False
        result = categorize(states[0], evals[0])
        assert result["reason"] == "low_reflection_parse_rate"

    return locals()
_suites['test_curation'] = _make_test_curation()
for _key, _fn in _suites['test_curation'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_curation__{_key}"] = _fn
del _make_test_curation


def _make_test_screening():
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

    return locals()
_suites['test_screening'] = _make_test_screening()
for _key, _fn in _suites['test_screening'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_screening__{_key}"] = _fn
del _make_test_screening


def _make_test_gates():
    import pytest
    from refldyn.gates import assess_train, build_gate, verify_gate
    from refldyn.io import save_jsonl, write_json
    row = _suites['test_metrics']['row']


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

    return locals()
_suites['test_gates'] = _make_test_gates()
for _key, _fn in _suites['test_gates'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_gates__{_key}"] = _fn
del _make_test_gates


def _make_test_answers():
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

    return locals()
_suites['test_answers'] = _make_test_answers()
for _key, _fn in _suites['test_answers'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_answers__{_key}"] = _fn
del _make_test_answers


def _make_test_corpus():
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

    return locals()
_suites['test_corpus'] = _make_test_corpus()
for _key, _fn in _suites['test_corpus'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_corpus__{_key}"] = _fn
del _make_test_corpus


def _make_test_protocol():
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

    return locals()
_suites['test_protocol'] = _make_test_protocol()
for _key, _fn in _suites['test_protocol'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_protocol__{_key}"] = _fn
del _make_test_protocol


def _make_test_selection():
    import pytest
    from refldyn.selection import build_states, ensure_isolation
    from refldyn.io import save_jsonl


    def drafts(split="train"):
        result = []
        for item, correct in enumerate([True, False, True, False]):
            result.append({"id": "p1", "split": split, "sample": item, "question": "3+4?", "gold": "7",
                           "source": "math", "model": "Qwen/Qwen2.5-32B", "initial_answer": "7" if correct else "8",
                           "trace": "work", "correct": correct, "valid": True})
        return result


    def test_states_true_model_trajectories():
        states, audit, report = build_states(drafts(), "train", minimum_samples=4)
        assert report["states"] == 2
        assert {x["initial_correct"] for x in states} == {True, False}
        assert all(x["draft_model"] == "Qwen/Qwen2.5-32B" for x in states)


    def test_wrong_split_rejected():
        with pytest.raises(ValueError):
            build_states(drafts("test"), "train")


    def test_split_isolation(tmp_path):
        a, _audit, _report = build_states(drafts(), "train", minimum_samples=4)
        save_jsonl(tmp_path / "train.jsonl", a)
        assert ensure_isolation({"train": tmp_path / "train.jsonl"})["unique_problems"] == 1
        save_jsonl(tmp_path / "test.jsonl", [{**x, "split": "test"} for x in a])
        with pytest.raises(ValueError, match="Leakage"):
            ensure_isolation({"train": tmp_path / "train.jsonl", "test": tmp_path / "test.jsonl"})

    return locals()
_suites['test_selection'] = _make_test_selection()
for _key, _fn in _suites['test_selection'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_selection__{_key}"] = _fn
del _make_test_selection


def _make_test_research_gate():
    import pytest

    from refldyn.gates import build_gate
    from refldyn.io import save_jsonl, checksum_file
    sample_row = _suites['test_metrics']['row']


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

    return locals()
_suites['test_research_gate'] = _make_test_research_gate()
for _key, _fn in _suites['test_research_gate'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_research_gate__{_key}"] = _fn
del _make_test_research_gate


def _make_test_training():
    import pytest
    from refldyn.train_grpo import prepare_training_records, outcome_reward
    from refldyn.checks import inspect_config
    state = _suites['test_gates']['state']


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

    return locals()
_suites['test_training'] = _make_test_training()
for _key, _fn in _suites['test_training'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_training__{_key}"] = _fn
del _make_test_training


def _make_test_offline_cli():
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from refldyn.io import load_jsonl, save_jsonl, write_json, checksum_file
    inputs = _suites['test_curation']['inputs']
    drafts = _suites['test_screening']['drafts']


    def call_cli(*args):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        return subprocess.run([sys.executable, "-m", "refldyn.cli", *map(str, args)],
                              check=True, env=env, capture_output=True, text=True)


    def test_screen_and_curate_file_interfaces(tmp_path):
        draft_file = tmp_path / "drafts.jsonl"
        save_jsonl(draft_file, drafts("a", correct_count=4) + drafts("b", correct_count=7))
        call_cli("screen", "--drafts", draft_file, "--split", "train", "--output-dir", tmp_path / "calibration")
        report = json.loads((tmp_path / "calibration/train.summary.json").read_text())
        assert report["eligible_problems"] == 2
        states, evals = inputs(80)
        candidates = tmp_path / "states.jsonl"
        probes = tmp_path / "probes.jsonl"
        training = tmp_path / "train.jsonl"
        save_jsonl(candidates, states)
        save_jsonl(probes, evals)
        write_json(str(probes) + ".manifest.json", {
            "states_sha256": checksum_file(candidates), "checkpoint": "base", "adapter": None,
            "retry_control": True, "model": "Qwen/Qwen2.5-32B"})
        call_cli("curate", "--candidate-states", candidates, "--probe-results", probes,
                 "--output", training, "--target", "40")
        chosen = list(load_jsonl(training))
        assert len(chosen) == 40
        manifest = json.loads((tmp_path / "train.curation.json").read_text())
        assert manifest["training_sha256"]

    return locals()
_suites['test_offline_cli'] = _make_test_offline_cli()
for _key, _fn in _suites['test_offline_cli'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_offline_cli__{_key}"] = _fn
del _make_test_offline_cli


def _make_test_resume():
    import json
    import sys

    import pytest

    from refldyn import evaluate, rollouts
    from refldyn.io import load_jsonl, save_jsonl
    FakeBackend = _suites['test_eval_backend']['FakeBackend']


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

    return locals()
_suites['test_resume'] = _make_test_resume()
for _key, _fn in _suites['test_resume'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_resume__{_key}"] = _fn
del _make_test_resume


def _make_test_trajectory():
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

    return locals()
_suites['test_trajectory'] = _make_test_trajectory()
for _key, _fn in _suites['test_trajectory'].items():
    if _key.startswith("test_") and callable(_fn):
        globals()[f"test_trajectory__{_key}"] = _fn
del _make_test_trajectory
