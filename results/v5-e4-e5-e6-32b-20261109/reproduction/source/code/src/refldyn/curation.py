import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .io import checksum_file, flatten, load_jsonl, save_jsonl, write_json


CATEGORIES = ("recoverable_error", "unresolved_error", "fragile_correct", "stable_correct")
DEFAULT_SHARES = {"recoverable_error": 0.35, "unresolved_error": 0.20,
                  "fragile_correct": 0.25, "stable_correct": 0.20}


def categorize(state, outcome, min_repetitions=4, min_valid_rate=0.75):
    if state["split"] != "train" or outcome["split"] != "train":
        raise ValueError("Curation is restricted to training data")
    if state["id"] != outcome["id"] or state["problem_id"] != outcome["problem_id"]:
        raise ValueError("Curation input state mismatch")
    if outcome["checkpoint"] != "base":
        raise ValueError("Only pre-training base-model probes may select training states")
    samples = outcome.get("reflect_samples", [])
    if len(samples) < min_repetitions:
        return {"eligible": False, "reason": "too_few_reflection_samples"}
    n_valid = sum(x.get("valid") is True for x in samples)
    valid_fraction = n_valid / len(samples)
    if valid_fraction < min_valid_rate:
        return {"eligible": False, "reason": "low_reflection_parse_rate",
                "forced_valid_rate": valid_fraction}
    n_correct = sum(x.get("correct") is True and x.get("valid") is True for x in samples)
    success = n_correct / len(samples)
    initial = bool(state["initial_correct"])
    category = ("fragile_correct" if success < 1 else "stable_correct") if initial else (
        "recoverable_error" if n_correct else "unresolved_error")
    return {"eligible": True, "category": category, "forced_accuracy": success,
            "forced_valid_rate": valid_fraction, "reflection_samples": len(samples),
            "empirical_gain": success - float(initial)}


def curate_training(states, evaluations, target=600, seed=41, max_per_problem=1,
                    shares=None, min_repetitions=4, min_valid_rate=0.75):
    if target < 1 or max_per_problem < 1:
        raise ValueError("Invalid curation size")
    shares = shares or DEFAULT_SHARES
    if set(shares) != set(CATEGORIES) or any(v < 0 for v in shares.values()) or abs(sum(shares.values()) - 1) > 1e-6:
        raise ValueError("Invalid curation category proportions")
    by_id = {}
    for row in states:
        if (row.get("split") != "train" or row.get("fixture")
                or row.get("source") == "synthetic_integer_arithmetic" or row.get("draft_adapter")) :
            raise ValueError("Only real training states may be curated")
        if row["id"] in by_id:
            raise ValueError(f"Duplicate state {row['id']}")
        by_id[row["id"]] = row
    model_ids = {r.get("draft_model") for r in states}
    if len(model_ids) != 1 or None in model_ids:
        raise ValueError("All candidates must use one identified draft model")
    results = {}
    for row in evaluations:
        if row.get("skip_reason"):
            continue
        if row["id"] in results:
            raise ValueError(f"Duplicate probe result {row['id']}")
        if row["id"] not in by_id:
            raise ValueError(f"Probe contains unknown state {row['id']}")
        if row.get("model") and row["model"] not in model_ids:
            raise ValueError("Probe model does not match the initial reasoning model")
        if row.get("adapter"):
            raise ValueError("Primary curation may not use an adapter-trained probe")
        results[row["id"]] = row
    rng = random.Random(seed)
    buckets = defaultdict(list)
    audit = Counter()
    for state_id, state in sorted(by_id.items()):
        outcome = results.get(state_id)
        if outcome is None:
            audit["unprobed"] += 1
            continue
        label = categorize(state, outcome, min_repetitions, min_valid_rate)
        if not label["eligible"]:
            audit[label["reason"]] += 1
            continue
        candidate = dict(state)
        candidate["curated_with"] = "base_forced_reflection"
        candidate["probe_category"] = label["category"]
        candidate["probe_reflection_samples"] = label["reflection_samples"]
        candidate["probe_empirical_gain"] = label["empirical_gain"]
        buckets[label["category"]].append(candidate)
    for name in CATEGORIES:
        rng.shuffle(buckets[name])
    chosen = []
    chosen_ids = set()
    per_problem = Counter()

    def consider(row):
        if row["id"] in chosen_ids or per_problem[row["problem_id"]] >= max_per_problem:
            return False
        chosen.append(row)
        chosen_ids.add(row["id"])
        per_problem[row["problem_id"]] += 1
        return True

    for category in CATEGORIES:
        quota = round(target * shares[category])
        selected = 0
        for row in buckets[category]:
            if selected >= quota:
                break
            if consider(row):
                selected += 1
    residual = [row for category in CATEGORIES for row in buckets[category] if row["id"] not in chosen_ids]
    rng.shuffle(residual)
    for row in residual:
        if len(chosen) >= target:
            break
        consider(row)
    chosen.sort(key=lambda x: x["id"])
    categories = Counter(row["probe_category"] for row in chosen)
    difficulty = Counter(row.get("baseline_bin", "unknown") for row in chosen)
    levels = Counter(row.get("level", "unknown") for row in chosen)
    report = {"requested_states": target, "curated_states": len(chosen),
              "unique_problems": len(per_problem), "available_candidates": dict((cat, len(buckets[cat])) for cat in CATEGORIES),
              "selected_categories": dict(categories), "selected_difficulty_bands": dict(difficulty),
              "selected_competition_levels": dict(levels), "discarded": dict(audit),
              "correct_fraction": sum(bool(r["initial_correct"]) for r in chosen) / len(chosen) if chosen else None,
              "seed": seed, "max_per_problem": max_per_problem, "shares": shares,
              "provenance": "selection from base-model forced-reflection probes on training source problems only",
              "sampling_note": "Probe-dependent training selection changes the training distribution; evaluation panels remain unselected."}
    if not chosen:
        raise ValueError("No curatable states. Inspect forced reflection format and sampling coverage")
    return chosen, report


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-states", required=True)
    parser.add_argument("--probe-results", nargs="+", required=True)
    parser.add_argument("--output", default="data/states/train.jsonl")
    parser.add_argument("--target", type=int, default=600)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-per-problem", type=int, default=1)
    parser.add_argument("--minimum-repetitions", type=int, default=4)
    parser.add_argument("--minimum-valid-rate", type=float, default=0.75)
    args = parser.parse_args()
    candidate_rows = list(load_jsonl(args.candidate_states))
    known_models = {r.get("draft_model") for r in candidate_rows}
    if len(known_models) != 1 or None in known_models:
        raise ValueError("Candidate file must declare a single base model")
    expected_model = known_models.pop()
    for path in args.probe_results:
        manifest_path = Path(str(path) + ".manifest.json")
        if not manifest_path.exists():
            raise ValueError(f"Missing base-model probe manifest: {manifest_path}")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("states_sha256") != checksum_file(args.candidate_states):
            raise ValueError("Probe manifest references another candidate-state file")
        if metadata.get("checkpoint") != "base" or metadata.get("adapter"):
            raise ValueError("Training curation requires a base-model probe")
        if metadata.get("retry_control") is not True:
            raise ValueError("Training curation requires the independent-retry control")
        if metadata.get("model") != expected_model:
            raise ValueError("Probe model does not match the source model")
    rows, report = curate_training(candidate_rows, flatten(args.probe_results),
                                   args.target, args.seed, args.max_per_problem,
                                   min_repetitions=args.minimum_repetitions,
                                   min_valid_rate=args.minimum_valid_rate)
    save_jsonl(args.output, rows)
    report.update({"candidate_states_sha256": checksum_file(args.candidate_states),
                   "probe_files_sha256": {path: checksum_file(path) for path in args.probe_results},
                   "training_sha256": checksum_file(args.output)})
    write_json(str(Path(args.output).with_suffix("")) + ".curation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
