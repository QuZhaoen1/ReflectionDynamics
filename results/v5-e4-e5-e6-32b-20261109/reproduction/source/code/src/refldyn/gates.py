import argparse
from pathlib import Path

from .io import checksum_file, load_jsonl, write_json
from .metrics import summarize


def assess_train(states, model, pilot=False):
    if not states:
        raise ValueError("No training states")
    if any(row.get("split") != "train" for row in states):
        raise ValueError("Non-training state in training input")
    if any(row.get("source") == "synthetic_integer_arithmetic" or row.get("fixture") for row in states):
        raise ValueError("Arithmetic fixtures are prohibited in the primary training run")
    if any(not row.get("initial_reasoning") or row.get("initial_answer") is None for row in states):
        raise ValueError("States require an actual model-generated reasoning trajectory")
    if any(row.get("draft_adapter") for row in states):
        raise ValueError("Primary RL training requires base-model initial trajectories")
    if any(row.get("draft_model") and row["draft_model"] != model for row in states):
        raise ValueError("Draft model differs from training model; record an explicit transfer experiment")
    unique = len(set(row["problem_id"] for row in states))
    correct = sum(row["initial_correct"] for row in states)
    balanced = 0.15 <= correct / len(states) <= 0.85
    return {"states": len(states), "unique_problems": unique, "correct_fraction": correct / len(states),
            "balanced": balanced, "size_pass": bool(pilot or (len(states) >= 300 and unique >= 200)),
            "split_pass": True, "real_trajectories": True}


def build_gate(train_states, base_eval, model, train_file, minimum_format=0.55,
               minimum_repair=0.03, pilot=False, validation_states=None, curation_report=None):
    training = assess_train(train_states, model, pilot)
    if not base_eval:
        raise ValueError("Missing baseline evaluation")
    if any(row.get("checkpoint") != "base" for row in base_eval):
        raise ValueError("Gate must use only pre-RL base-model results")
    if any(row.get("split") != "validation" for row in base_eval):
        raise ValueError("Gate must use validation only, never test or OOD")
    baseline_results = [r for r in base_eval if not r.get("skip_reason")]
    if not baseline_results:
        raise ValueError("No valid baseline evaluation results")
    summary = summarize(baseline_results)
    unique_eval = summary["unique_problems"]
    validation_coverage = None
    if validation_states is not None:
        expected = {r["id"] for r in validation_states}
        reported = [r["id"] for r in base_eval]
        if len(reported) != len(set(reported)):
            raise ValueError("Duplicate baseline evaluation state")
        if set(reported) != expected:
            raise ValueError("Base evaluation does not match the fixed validation panel")
        if set(r["problem_id"] for r in train_states) & {r["problem_id"] for r in validation_states}:
            raise ValueError("Train/validation problem leakage")
        validation_coverage = len(baseline_results) / len(expected)
    if len({r["id"] for r in train_states}) != len(train_states):
        raise ValueError("Duplicate training state ID")
    curated_ok = curation_report is not None and all(
        r.get("curated_with") == "base_forced_reflection" for r in train_states)
    if curation_report is not None:
        curated_ok = bool(curated_ok and curation_report.get("training_sha256") == checksum_file(train_file)
                          and curation_report.get("curated_states") == len(train_states))
    recoverable_count = sum(r.get("probe_category") == "recoverable_error" for r in train_states)
    correct_control_count = sum(r.get("initial_correct") is True for r in train_states)
    result = {
        "training": training, "base_evaluation": summary,
        "model": model, "train_file_sha256": checksum_file(train_file),
        "requirements": {"min_train_states": 300, "min_train_problems": 200,
                         "min_validation_problems": 40, "min_natural_format": minimum_format,
                         "min_forced_format": minimum_format,
                         "min_repair": minimum_repair, "min_coverage": 0.95,
                         "min_recoverable_states": 20, "min_correct_control_states": 40},
    }
    conditions = {
        "train_size": training["size_pass"], "train_balance": training["balanced"],
        "validation_size": bool(pilot or unique_eval >= 40),
        "natural_format": summary["natural_format_rate"] >= minimum_format,
        "forced_format": summary["forced_format_rate"] >= minimum_format,
        "repair_nonzero": summary["repair_rate"] is not None and summary["repair_rate"] >= minimum_repair,
        "validation_coverage": bool(validation_coverage is None or validation_coverage >= 0.95),
        "trained_on_probed_states": bool(curation_report is None or curated_ok),
        "recoverable_cases": bool(curation_report is None or recoverable_count >= 20),
        "correct_controls": bool(curation_report is None or correct_control_count >= 40),
    }
    result["validation_coverage"] = validation_coverage
    result["curation_report_verified"] = curated_ok
    result["recoverable_training_cases"] = recoverable_count
    result["correct_training_controls"] = correct_control_count
    result["conditions"] = conditions
    result["passed"] = all(conditions.values())
    result["research_status"] = "pilot_only" if pilot else "baseline_gate"
    return result


def verify_gate(path, model, train_file):
    import json
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not report.get("passed") or report.get("research_status") != "baseline_gate":
        raise ValueError("Baseline gate was not passed for full training")
    if report.get("model") != model:
        raise ValueError("Gate model does not match training model")
    if report.get("train_file_sha256") != checksum_file(train_file):
        raise ValueError("Training data changed since baseline gate")
    return True


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-states", required=True)
    parser.add_argument("--validation-results", required=True, nargs="+")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--output", default="data/selection/train_gate.json")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--validation-states", required=True)
    parser.add_argument("--curation-report", required=True)
    args = parser.parse_args()
    from .io import flatten
    import json
    validation = list(load_jsonl(args.validation_states))
    for path in args.validation_results:
        manifest_file = Path(str(path) + ".manifest.json")
        if not manifest_file.exists():
            raise ValueError(f"Missing evaluation manifest: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("states_sha256") != checksum_file(args.validation_states):
            raise ValueError("Evaluation manifest was generated from a different panel")
        if manifest.get("model") != args.model or manifest.get("checkpoint") != "base":
            raise ValueError("Evaluation manifest has the wrong model or checkpoint")
    curation = json.loads(Path(args.curation_report).read_text(encoding="utf-8"))
    result = build_gate(list(load_jsonl(args.train_states)), flatten(args.validation_results),
                        args.model, args.train_states, pilot=args.pilot,
                        validation_states=validation, curation_report=curation)
    write_json(args.output, result)
    print("PASS" if result["passed"] else "STOP")
    for name, value in result["conditions"].items():
        print(f"  {name}: {'pass' if value else 'fail'}")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    run()
